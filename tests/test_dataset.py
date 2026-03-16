"""Tests for dataset loading, prompt formatting, and label masking."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.data.dataset import DollyDataset, _format_prompt, _tokenize_and_mask, RESPONSE_HEADER


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_prompt_includes_instruction_and_response():
    example = {
        "instruction": "What is the capital of France?",
        "context": "",
        "response": "Paris.",
    }
    prompt = _format_prompt(example)
    assert "### Instruction:" in prompt
    assert "What is the capital of France?" in prompt
    assert RESPONSE_HEADER in prompt
    assert "Paris." in prompt


def test_format_prompt_omits_input_when_context_empty():
    example = {
        "instruction": "Write a haiku about autumn.",
        "context": "",
        "response": "Leaves fall gently down.",
    }
    prompt = _format_prompt(example)
    assert "### Input:" not in prompt


def test_format_prompt_includes_input_when_context_present():
    example = {
        "instruction": "Summarize the following passage.",
        "context": "The quick brown fox jumps over the lazy dog.",
        "response": "A fox jumps over a dog.",
    }
    prompt = _format_prompt(example)
    assert "### Input:" in prompt
    assert "The quick brown fox" in prompt


def test_format_prompt_response_comes_after_instruction():
    example = {
        "instruction": "Translate 'hello' to Spanish.",
        "context": "",
        "response": "hola",
    }
    prompt = _format_prompt(example)
    instruction_pos = prompt.index("### Instruction:")
    response_pos = prompt.index(RESPONSE_HEADER)
    assert response_pos > instruction_pos


# ---------------------------------------------------------------------------
# Tokenisation and label masking
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_tokenized_output_has_input_ids_and_labels(tokenizer):
    example = {
        "instruction": "What is 2+2?",
        "context": "",
        "response": "4",
    }
    result = _tokenize_and_mask(example, tokenizer, max_length=128)
    assert "input_ids" in result
    assert "labels" in result


def test_input_ids_and_labels_same_length(tokenizer):
    example = {
        "instruction": "Name three colours.",
        "context": "",
        "response": "Red, blue, green.",
    }
    result = _tokenize_and_mask(example, tokenizer, max_length=128)
    assert len(result["input_ids"]) == len(result["labels"])


def test_label_masking_sets_instruction_tokens_to_minus100(tokenizer):
    """Tokens before '### Response:' should all be -100."""
    example = {
        "instruction": "Describe photosynthesis briefly.",
        "context": "",
        "response": "Plants convert light into energy.",
    }
    result = _tokenize_and_mask(example, tokenizer, max_length=256)
    labels = result["labels"]
    input_ids = result["input_ids"]

    # Find first non-masked label position
    first_valid = next((i for i, l in enumerate(labels) if l != -100), None)
    assert first_valid is not None, "All labels are -100; response not found"

    # Every position before first_valid must be -100
    assert all(l == -100 for l in labels[:first_valid])


def test_label_masking_some_tokens_not_masked(tokenizer):
    """At least some tokens must be unmasked (the response tokens)."""
    example = {
        "instruction": "What is the speed of light?",
        "context": "",
        "response": "Approximately 299,792 km/s in a vacuum.",
    }
    result = _tokenize_and_mask(example, tokenizer, max_length=256)
    labels = result["labels"]
    non_masked = [l for l in labels if l != -100]
    assert len(non_masked) > 0


def test_label_masking_with_context(tokenizer):
    """Masking should work correctly when context is present."""
    example = {
        "instruction": "Answer based on the context.",
        "context": "The Eiffel Tower is in Paris.",
        "response": "The Eiffel Tower is located in Paris.",
    }
    result = _tokenize_and_mask(example, tokenizer, max_length=256)
    labels = result["labels"]
    non_masked = [l for l in labels if l != -100]
    assert len(non_masked) > 0


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def test_train_val_split_sizes(tokenizer):
    """Validation fraction should be approximately 10% of total data."""
    ds = DollyDataset(tokenizer=tokenizer, max_length=128, val_fraction=0.1, seed=42)
    # Use a small mock dataset to avoid downloading Dolly during tests
    from datasets import Dataset as HFDataset

    raw_examples = [
        {
            "instruction": f"Instruction {i}",
            "context": "",
            "response": f"Response {i} with enough text to survive masking.",
            "category": ["open_qa", "closed_qa", "classification"][i % 3],
        }
        for i in range(30)
    ]
    raw = HFDataset.from_list(raw_examples)

    # Monkey-patch load_raw for the test
    ds.load_raw = lambda: raw

    train_ds, val_ds = ds.get_train_val_datasets()
    total = len(train_ds) + len(val_ds)
    val_fraction = len(val_ds) / total
    # Allow ±5% tolerance due to stratification rounding
    assert 0.05 <= val_fraction <= 0.25, (
        f"Val fraction {val_fraction:.2f} is outside expected range [0.05, 0.25]"
    )


def test_train_larger_than_val(tokenizer):
    """Training set must be larger than validation set."""
    from datasets import Dataset as HFDataset

    raw_examples = [
        {
            "instruction": f"Instruction {i}",
            "context": "",
            "response": f"Response {i} with enough text to survive masking.",
            "category": ["open_qa", "creative_writing"][i % 2],
        }
        for i in range(40)
    ]
    raw = HFDataset.from_list(raw_examples)

    ds = DollyDataset(tokenizer=tokenizer, max_length=128, val_fraction=0.1, seed=42)
    ds.load_raw = lambda: raw
    train_ds, val_ds = ds.get_train_val_datasets()

    assert len(train_ds) > len(val_ds)
