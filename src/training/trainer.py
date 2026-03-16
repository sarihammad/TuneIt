"""MethodTrainer orchestrates the full fine-tuning/LoRA/QLoRA training loop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import LoRAConfig, ModelConfig, QLoRAConfig, TrainConfig
from src.data.collator import build_data_collator
from src.data.dataset import DollyDataset
from src.models.base import ModelLoader
from src.models.lora import LoRAAdapter
from src.models.qlora import QLoRAAdapter
from src.training.arguments import get_training_arguments
from src.training.callbacks import MemoryTrackingCallback, ThroughputCallback

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer


@dataclass
class TrainingResult:
    """Structured output from a single training run."""

    method: str
    train_loss: float
    eval_loss: float
    perplexity: float
    peak_memory_mb: float
    mean_throughput_tokens_per_sec: float
    trainable_params: int
    total_params: int
    trainable_fraction: float
    training_time_seconds: float
    checkpoint_path: str
    extra_metrics: Dict[str, float] = field(default_factory=dict)


class MethodTrainer:
    """Unified trainer that supports full fine-tuning, LoRA, and QLoRA.

    Usage:
        trainer = MethodTrainer(
            method="lora",
            model_config=ModelConfig(model_name="facebook/opt-125m"),
            adapter_config=LoRAConfig(),
            train_config=TrainConfig(method="lora"),
        )
        trainer.setup()
        result = trainer.train()
    """

    def __init__(
        self,
        method: str,
        model_config: ModelConfig,
        train_config: TrainConfig,
        adapter_config: Optional[LoRAConfig | QLoRAConfig] = None,
    ) -> None:
        if method not in {"full", "lora", "qlora"}:
            raise ValueError(f"Unknown method {method!r}. Choose full/lora/qlora.")
        self.method = method
        self.model_config = model_config
        self.train_config = train_config
        self.adapter_config = adapter_config

        self.model: Optional[PreTrainedModel | PeftModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        self._memory_cb: Optional[MemoryTrackingCallback] = None
        self._throughput_cb: Optional[ThroughputCallback] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load model, apply adapters, and prepare datasets."""
        self._load_model()
        self._load_datasets()

    def _load_model(self) -> None:
        mc = self.model_config

        if self.method == "full":
            model, tokenizer = ModelLoader.load_full(
                mc.model_name,
                torch_dtype=mc.torch_dtype,
                device_map=mc.device_map,
            )

        elif self.method == "lora":
            model, tokenizer = ModelLoader.load_for_lora(
                mc.model_name,
                torch_dtype=mc.torch_dtype,
                device_map=mc.device_map,
            )
            cfg = self.adapter_config if isinstance(self.adapter_config, LoRAConfig) else LoRAConfig()
            model = LoRAAdapter.apply(model, cfg)

        elif self.method == "qlora":
            model, tokenizer = ModelLoader.load_for_qlora(
                mc.model_name,
                device_map=mc.device_map,
            )
            cfg = (
                self.adapter_config
                if isinstance(self.adapter_config, QLoRAConfig)
                else QLoRAConfig()
            )
            model = QLoRAAdapter.apply(model, cfg)
            QLoRAAdapter.apply_gradient_checkpointing(model)

        self.model = model
        self.tokenizer = tokenizer

    def _load_datasets(self) -> None:
        dolly = DollyDataset(
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_length,
            seed=self.train_config.seed,
        )
        self.train_dataset, self.eval_dataset = dolly.get_train_val_datasets()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run the training loop and return structured metrics."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call setup() before train().")

        param_info = ModelLoader.count_parameters(self.model)
        collator = build_data_collator(self.tokenizer, self.model)
        training_args = get_training_arguments(self.train_config)

        self._memory_cb = MemoryTrackingCallback(log_every_n_steps=10)
        self._throughput_cb = ThroughputCallback()

        hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            callbacks=[self._memory_cb, self._throughput_cb],
        )

        t0 = time.perf_counter()
        train_output = hf_trainer.train()
        training_time = time.perf_counter() - t0

        eval_metrics = hf_trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("nan"))
        perplexity = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")

        checkpoint_path = training_args.output_dir
        hf_trainer.save_model(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        return TrainingResult(
            method=self.method,
            train_loss=train_output.training_loss,
            eval_loss=eval_loss,
            perplexity=perplexity,
            peak_memory_mb=self._memory_cb.peak_memory_mb(),
            mean_throughput_tokens_per_sec=self._throughput_cb.mean_throughput(),
            trainable_params=param_info["trainable_params"],
            total_params=param_info["total_params"],
            trainable_fraction=param_info["trainable_fraction"],
            training_time_seconds=training_time,
            checkpoint_path=checkpoint_path,
        )
