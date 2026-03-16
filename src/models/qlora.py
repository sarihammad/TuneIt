"""QLoRA adapter: 4-bit NF4 quantisation + LoRA, with gradient checkpointing."""

from __future__ import annotations

from src.config import QLoRAConfig

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel


class QLoRAAdapter:
    """Applies LoRA adapters on top of a 4-bit quantised base model.

    The base model must already be loaded via ModelLoader.load_for_qlora()
    which handles BitsAndBytesConfig and enable_input_require_grads().
    This class attaches the trainable low-rank adapter matrices and
    optionally activates gradient checkpointing to trade compute for memory.
    """

    @staticmethod
    def apply(model: PreTrainedModel, config: QLoRAConfig) -> PeftModel:
        """Inject LoRA adapters onto a quantised model.

        Differences from standard LoRA:
        - use_cache must be False (set in load_for_qlora; gradient
          checkpointing and KV-cache are mutually exclusive).
        - Broader target_modules coverage (q,k,v,o by default) compensates
          for representation capacity lost to 4-bit quantisation.
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
    def apply_gradient_checkpointing(model: PeftModel) -> None:
        """Enable gradient checkpointing on the underlying base model.

        Gradient checkpointing recomputes activations during the backward
        pass instead of storing them, reducing activation memory by roughly
        sqrt(sequence_length) at the cost of ~33% extra FLOPs.

        Must be called AFTER get_peft_model() because PEFT wraps the model
        and the gradient-checkpoint hook needs to be on the outermost module.
        """
        model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model.base_model, "gradient_checkpointing_enable"):
            model.base_model.gradient_checkpointing_enable()
