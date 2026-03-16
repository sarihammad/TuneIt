"""Base model loading: full precision and 4-bit quantized variants."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unknown torch_dtype {dtype_str!r}. Choose from {list(mapping)}"
        )
    return mapping[dtype_str]


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Many causal LMs ship without a dedicated pad token; reuse eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class ModelLoader:
    """Factory for loading base models in various precision modes."""

    @staticmethod
    def load_full(
        model_name: str,
        torch_dtype: str = "float32",
        device_map: str = "auto",
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model in full precision for standard fine-tuning.

        Gradient checkpointing is recommended on top of this to reduce
        activation memory during full fine-tuning.
        """
        dtype = _get_torch_dtype(torch_dtype)
        tokenizer = _load_tokenizer(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model.config.use_cache = False
        return model, tokenizer

    @staticmethod
    def load_for_qlora(
        model_name: str,
        device_map: str = "auto",
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model in 4-bit NF4 quantisation for QLoRA fine-tuning.

        BitsAndBytesConfig parameters:
        - bnb_4bit_compute_dtype=bfloat16: dequantise to BF16 for matrix
          multiplication; keeps numerical stability while saving memory.
        - bnb_4bit_use_double_quant=True: quantise the quantisation
          constants themselves, saving ~0.4 bits/param additionally.
        - bnb_4bit_quant_type="nf4": Normal Float 4 is information-
          theoretically optimal for normally distributed weights, unlike
          INT4 which assumes a uniform distribution.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = _load_tokenizer(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        # Disable KV-cache; incompatible with gradient checkpointing.
        model.config.use_cache = False

        # Required before apply_gradient_checkpointing so that gradients
        # flow through the frozen quantised layers to the LoRA adapters.
        model.enable_input_require_grads()

        return model, tokenizer

    @staticmethod
    def load_for_lora(
        model_name: str,
        torch_dtype: str = "float32",
        device_map: str = "auto",
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model in full (or half) precision for LoRA fine-tuning."""
        return ModelLoader.load_full(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )

    @staticmethod
    def count_parameters(model: PreTrainedModel) -> Dict[str, int | float]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        fraction = trainable / total if total > 0 else 0.0
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "trainable_fraction": fraction,
        }
