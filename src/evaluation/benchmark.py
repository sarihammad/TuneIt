"""Benchmark evaluator: perplexity, ROUGE, inference latency, memory footprint."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


_FIXED_LATENCY_PROMPT = (
    "### Instruction:\nExplain the difference between supervised and "
    "unsupervised learning in machine learning.\n\n### Response:\n"
)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics for a single model."""

    method: str
    perplexity: float
    rouge1: float
    rouge2: float
    rougeL: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float
    model_size_mb: float
    peak_inference_memory_mb: float


class BenchmarkEvaluator:
    """Runs a standardised set of benchmarks for comparing fine-tuning methods."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        method: str = "unknown",
        device: str = "auto",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.method = method

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def compute_perplexity(
        self,
        dataset: Dataset,
        max_samples: int = 200,
    ) -> float:
        """Compute perplexity on a held-out set.

        Perplexity = exp(mean cross-entropy loss) evaluated with the model
        in eval mode. Only response tokens contribute to the loss because
        the dataset was tokenized with instruction-masking.
        """
        self.model.eval()
        losses: List[float] = []
        n = min(max_samples, len(dataset))

        with torch.no_grad():
            for i in tqdm(range(n), desc="Computing perplexity", leave=False):
                example = dataset[i]
                input_ids = torch.tensor(
                    [example["input_ids"]], dtype=torch.long
                ).to(self.device)
                labels = torch.tensor(
                    [example["labels"]], dtype=torch.long
                ).to(self.device)

                try:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    if loss is not None and not torch.isnan(loss):
                        losses.append(loss.item())
                except Exception:
                    continue

        if not losses:
            return float("nan")
        mean_loss = sum(losses) / len(losses)
        return math.exp(mean_loss)

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------

    def compute_rouge(
        self,
        dataset: Dataset,
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """Generate responses and compute ROUGE against references.

        The model is prompted with the instruction only; the generated
        output is compared against the reference response from the dataset.
        """
        self.model.eval()
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        n = min(max_samples, len(dataset))

        for i in tqdm(range(n), desc="Computing ROUGE", leave=False):
            example = dataset[i]
            input_ids = example["input_ids"]
            labels = example["labels"]

            # Reconstruct the instruction prefix (all -100 masked positions)
            prompt_ids = [
                tid for tid, lbl in zip(input_ids, labels) if lbl == -100
            ]
            if not prompt_ids:
                continue

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(
                self.device
            )

            with torch.no_grad():
                try:
                    generated = self.model.generate(
                        prompt_tensor,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except Exception:
                    continue

            # Decode only newly generated tokens
            new_tokens = generated[0][len(prompt_ids):]
            prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Reference: decode the non-masked label tokens
            ref_ids = [lbl for lbl in labels if lbl != -100]
            reference = self.tokenizer.decode(ref_ids, skip_special_tokens=True)

            if not reference.strip():
                continue

            result = scorer.score(reference, prediction)
            scores["rouge1"].append(result["rouge1"].fmeasure)
            scores["rouge2"].append(result["rouge2"].fmeasure)
            scores["rougeL"].append(result["rougeL"].fmeasure)

        return {
            k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()
        }

    # ------------------------------------------------------------------
    # Inference latency
    # ------------------------------------------------------------------

    def measure_inference_latency(self, n_runs: int = 50) -> Dict[str, float]:
        """Measure wall-clock inference time on a fixed prompt.

        A warm-up pass is performed before timing to avoid measuring
        one-time JIT compilation or CUDA kernel launch overhead.
        """
        self.model.eval()
        inputs = self.tokenizer(
            _FIXED_LATENCY_PROMPT,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies: List[float] = []
        for _ in tqdm(range(n_runs), desc="Measuring latency", leave=False):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

        latencies_arr = np.array(latencies)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out.shape[1] - prompt_len
        tokens_per_sec = (new_tokens * 1000.0) / float(np.mean(latencies_arr))

        return {
            "mean_ms": float(np.mean(latencies_arr)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "tokens_per_second": tokens_per_sec,
        }

    # ------------------------------------------------------------------
    # Memory footprint
    # ------------------------------------------------------------------

    def measure_memory_footprint(self) -> Dict[str, float]:
        """Estimate model weight memory and peak inference memory.

        Uses model.get_memory_footprint() when available (PEFT models expose
        this). Falls back to counting bytes per parameter.
        """
        if hasattr(self.model, "get_memory_footprint"):
            model_bytes = self.model.get_memory_footprint()
        else:
            dtype_bytes = {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
            }
            total_bytes = 0
            for p in self.model.parameters():
                nbytes = dtype_bytes.get(p.dtype, 4)
                total_bytes += p.numel() * nbytes
            model_bytes = total_bytes

        model_size_mb = model_bytes / (1024**2)

        # Measure peak inference memory by running a forward pass.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            inputs = self.tokenizer(
                _FIXED_LATENCY_PROMPT,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)
            self.model.eval()
            with torch.no_grad():
                self.model(**inputs)
            peak_inference_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_inference_mb = model_size_mb  # conservative fallback

        return {
            "model_size_mb": model_size_mb,
            "peak_inference_memory_mb": peak_inference_mb,
        }

    # ------------------------------------------------------------------
    # Composite run
    # ------------------------------------------------------------------

    def run_all(
        self,
        eval_dataset: Dataset,
        n_perplexity_samples: int = 200,
        n_rouge_samples: int = 100,
        n_latency_runs: int = 50,
    ) -> BenchmarkResult:
        """Run every benchmark and return a BenchmarkResult dataclass."""
        ppl = self.compute_perplexity(eval_dataset, max_samples=n_perplexity_samples)
        rouge = self.compute_rouge(eval_dataset, max_samples=n_rouge_samples)
        latency = self.measure_inference_latency(n_runs=n_latency_runs)
        memory = self.measure_memory_footprint()

        return BenchmarkResult(
            method=self.method,
            perplexity=ppl,
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
            mean_latency_ms=latency["mean_ms"],
            p50_latency_ms=latency["p50_ms"],
            p95_latency_ms=latency["p95_ms"],
            p99_latency_ms=latency["p99_ms"],
            tokens_per_second=latency["tokens_per_second"],
            model_size_mb=memory["model_size_mb"],
            peak_inference_memory_mb=memory["peak_inference_memory_mb"],
        )
