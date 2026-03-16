"""Tests for LoRA adapter application and parameter reduction."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def base_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.config.use_cache = False
    return model, tokenizer


@pytest.fixture(scope="module")
def lora_model(base_model_and_tokenizer):
    from src.config import LoRAConfig
    from src.models.lora import LoRAAdapter

    model, tokenizer = base_model_and_tokenizer
    cfg = LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    peft_model = LoRAAdapter.apply(model, cfg)
    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Parameter reduction
# ---------------------------------------------------------------------------


def test_lora_reduces_trainable_parameters(lora_model):
    """LoRA should have far fewer trainable params than total params."""
    from src.models.base import ModelLoader

    peft_model, _ = lora_model
    info = ModelLoader.count_parameters(peft_model)
    assert info["trainable_params"] < info["total_params"]
    assert info["trainable_params"] > 0


def test_lora_trainable_fraction_below_two_percent(lora_model):
    """For opt-125m with r=8 on q/v only, trainable fraction < 2%."""
    from src.models.base import ModelLoader

    peft_model, _ = lora_model
    info = ModelLoader.count_parameters(peft_model)
    assert info["trainable_fraction"] < 0.02, (
        f"Trainable fraction {info['trainable_fraction']:.4f} exceeds 2% threshold"
    )


def test_lora_target_modules_are_adapted(lora_model):
    """LoRA should have adapter parameters on the specified target modules."""
    peft_model, _ = lora_model
    adapter_param_names = [
        name for name, _ in peft_model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    assert len(adapter_param_names) > 0, "No LoRA adapter parameters found"
    # Check both q_proj and v_proj got adapters
    has_q = any("q_proj" in n for n in adapter_param_names)
    has_v = any("v_proj" in n for n in adapter_param_names)
    assert has_q, "q_proj not adapted"
    assert has_v, "v_proj not adapted"


def test_lora_non_target_modules_frozen(lora_model):
    """Non-adapter parameters must not require gradients."""
    peft_model, _ = lora_model
    for name, param in peft_model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            assert not param.requires_grad, (
                f"Non-adapter param {name} has requires_grad=True"
            )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_lora_forward_pass(lora_model):
    """Model with LoRA adapters should produce valid logits."""
    peft_model, tokenizer = lora_model
    inputs = tokenizer(
        "The meaning of life is", return_tensors="pt", truncation=True, max_length=32
    )
    with torch.no_grad():
        outputs = peft_model(**inputs)
    assert outputs.logits is not None
    assert outputs.logits.shape[-1] > 0  # vocab dimension
    assert not torch.isnan(outputs.logits).any()


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_lora_save_load_roundtrip(lora_model):
    """Saved adapter weights should match loaded adapter weights exactly."""
    from src.models.lora import LoRAAdapter

    peft_model, tokenizer = lora_model

    with tempfile.TemporaryDirectory() as tmpdir:
        peft_model.save_pretrained(tmpdir)

        # Load a fresh base model and attach saved adapter
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        loaded = LoRAAdapter.load_adapter(base, tmpdir)

        # Compare adapter weight tensors
        original_state = {
            k: v for k, v in peft_model.state_dict().items()
            if "lora_" in k
        }
        loaded_state = {
            k: v for k, v in loaded.state_dict().items()
            if "lora_" in k
        }

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key]), (
                f"Mismatch in adapter weight: {key}"
            )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_lora_config_validates_rank():
    from src.config import LoRAConfig
    import pydantic

    with pytest.raises((ValueError, pydantic.ValidationError)):
        LoRAConfig(r=-1)


def test_lora_config_defaults():
    from src.config import LoRAConfig

    cfg = LoRAConfig()
    assert cfg.r == 16
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules
    assert cfg.bias == "none"
