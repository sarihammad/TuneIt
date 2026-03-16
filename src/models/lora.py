"""LoRA adapter application and loading via the PEFT library."""

from __future__ import annotations

from src.config import LoRAConfig
from src.models.base import ModelLoader

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class LoRAAdapter:
    """Wraps PEFT LoRA operations: apply, inspect, and load adapters."""

    @staticmethod
    def apply(model: PreTrainedModel, config: LoRAConfig) -> PeftModel:
        """Inject LoRA adapters into the model's attention projections.

        Only the adapter weights (A and B matrices) are marked as trainable.
        The base weights remain frozen, reducing GPU memory and allowing
        much larger effective batch sizes.
        """
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        task_type = task_type_map.get(config.task_type, TaskType.CAUSAL_LM)

        lora_cfg = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=task_type,
        )

        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()
        return peft_model

    @staticmethod
    def load_adapter(base_model: PreTrainedModel, adapter_path: str) -> PeftModel:
        """Load a saved LoRA adapter checkpoint onto a base model.

        The adapter weights are merged onto the base model's weight tensors
        in memory; the original quantised or full-precision base weights are
        preserved and only the delta (B @ A) is added at inference time.
        """
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        return peft_model

    @staticmethod
    def load_full_pipeline(
        model_name: str,
        adapter_path: str,
        torch_dtype: str = "float32",
        device_map: str = "auto",
    ) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        """Convenience: load base model + adapter in one call."""
        model, tokenizer = ModelLoader.load_full(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        peft_model = LoRAAdapter.load_adapter(model, adapter_path)
        return peft_model, tokenizer
