"""Tests for benchmark evaluation: perplexity, latency, ROUGE, comparison."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def small_eval_dataset(model_and_tokenizer):
    """Create a small tokenized eval dataset without downloading Dolly."""
    from src.data.dataset import _tokenize_and_mask

    _, tokenizer = model_and_tokenizer
    examples = [
        {
            "instruction": f"What is {i} plus {i}?",
            "context": "",
            "response": f"The answer is {i * 2}.",
        }
        for i in range(1, 11)
    ]
    tokenized = [_tokenize_and_mask(ex, tokenizer, max_length=128) for ex in examples]
    # Keep only examples where some labels are not -100
    tokenized = [ex for ex in tokenized if any(l != -100 for l in ex["labels"])]
    return HFDataset.from_list(tokenized)


@pytest.fixture(scope="module")
def evaluator(model_and_tokenizer):
    from src.evaluation.benchmark import BenchmarkEvaluator

    model, tokenizer = model_and_tokenizer
    return BenchmarkEvaluator(model=model, tokenizer=tokenizer, method="test")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def test_perplexity_returns_positive_float(evaluator, small_eval_dataset):
    ppl = evaluator.compute_perplexity(small_eval_dataset, max_samples=5)
    assert isinstance(ppl, float)
    assert ppl > 0.0


def test_perplexity_is_finite(evaluator, small_eval_dataset):
    import math
    ppl = evaluator.compute_perplexity(small_eval_dataset, max_samples=5)
    assert math.isfinite(ppl)


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------


def test_latency_returns_correct_keys(evaluator):
    result = evaluator.measure_inference_latency(n_runs=3)
    required_keys = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "tokens_per_second"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_latency_values_are_positive(evaluator):
    result = evaluator.measure_inference_latency(n_runs=3)
    for key, val in result.items():
        assert val > 0.0, f"Latency metric {key}={val} should be positive"


def test_latency_percentiles_ordered(evaluator):
    result = evaluator.measure_inference_latency(n_runs=5)
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]


# ---------------------------------------------------------------------------
# ROUGE scores
# ---------------------------------------------------------------------------


def test_rouge_scores_between_zero_and_one(evaluator, small_eval_dataset):
    rouge = evaluator.compute_rouge(small_eval_dataset, max_samples=5)
    for key in ["rouge1", "rouge2", "rougeL"]:
        assert key in rouge, f"Missing ROUGE key: {key}"
        assert 0.0 <= rouge[key] <= 1.0, (
            f"ROUGE score {key}={rouge[key]} out of [0, 1]"
        )


def test_rouge_returns_all_variants(evaluator, small_eval_dataset):
    rouge = evaluator.compute_rouge(small_eval_dataset, max_samples=3)
    assert "rouge1" in rouge
    assert "rouge2" in rouge
    assert "rougeL" in rouge


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------


def test_memory_footprint_returns_correct_keys(evaluator):
    result = evaluator.measure_memory_footprint()
    assert "model_size_mb" in result
    assert "peak_inference_memory_mb" in result


def test_model_size_positive(evaluator):
    result = evaluator.measure_memory_footprint()
    assert result["model_size_mb"] > 0.0


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def test_comparison_table_structure():
    """Comparison table should have one row per method and expected columns."""
    from dataclasses import dataclass
    from src.evaluation.comparison import ComparisonBuilder
    from src.evaluation.benchmark import BenchmarkResult
    from src.training.trainer import TrainingResult

    def _make_training(method):
        return TrainingResult(
            method=method,
            train_loss=2.0,
            eval_loss=2.1,
            perplexity=8.16,
            peak_memory_mb=500.0,
            mean_throughput_tokens_per_sec=1000.0,
            trainable_params=800_000,
            total_params=125_000_000,
            trainable_fraction=0.0064,
            training_time_seconds=120.0,
            checkpoint_path=f"outputs/{method}",
        )

    def _make_benchmark(method):
        return BenchmarkResult(
            method=method,
            perplexity=8.16,
            rouge1=0.25,
            rouge2=0.10,
            rougeL=0.20,
            mean_latency_ms=50.0,
            p50_latency_ms=48.0,
            p95_latency_ms=60.0,
            p99_latency_ms=65.0,
            tokens_per_second=150.0,
            model_size_mb=480.0,
            peak_inference_memory_mb=512.0,
        )

    builder = ComparisonBuilder()
    for method in ["full", "lora", "qlora"]:
        builder.add_result(method, _make_training(method), _make_benchmark(method))

    df = builder.build_table()
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    assert "Method" in df.columns
    assert "Perplexity" in df.columns
    assert "ROUGE-L" in df.columns
    assert "Peak Train Memory (MB)" in df.columns
    assert set(df["Method"]) == {"full", "lora", "qlora"}


def test_comparison_save_creates_files(tmp_path):
    """save_report should create both .csv and .json files."""
    from src.evaluation.comparison import ComparisonBuilder
    from src.evaluation.benchmark import BenchmarkResult
    from src.training.trainer import TrainingResult

    builder = ComparisonBuilder()
    tr = TrainingResult(
        method="lora", train_loss=2.0, eval_loss=2.1, perplexity=8.16,
        peak_memory_mb=500.0, mean_throughput_tokens_per_sec=1000.0,
        trainable_params=800_000, total_params=125_000_000,
        trainable_fraction=0.0064, training_time_seconds=120.0,
        checkpoint_path="outputs/lora",
    )
    bm = BenchmarkResult(
        method="lora", perplexity=8.16, rouge1=0.25, rouge2=0.10,
        rougeL=0.20, mean_latency_ms=50.0, p50_latency_ms=48.0,
        p95_latency_ms=60.0, p99_latency_ms=65.0, tokens_per_second=150.0,
        model_size_mb=480.0, peak_inference_memory_mb=512.0,
    )
    builder.add_result("lora", tr, bm)

    base_path = str(tmp_path / "report")
    builder.save_report(base_path)

    assert os.path.exists(f"{base_path}.csv")
    assert os.path.exists(f"{base_path}.json")
