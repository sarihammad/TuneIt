"""Comparison report: aggregate results from multiple fine-tuning runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.evaluation.benchmark import BenchmarkResult
from src.training.trainer import TrainingResult


@dataclass
class MethodRecord:
    """Paired training and benchmark results for a single method."""

    method: str
    training: TrainingResult
    benchmark: BenchmarkResult


class ComparisonBuilder:
    """Collects results from multiple methods and renders comparison reports."""

    def __init__(self) -> None:
        self._records: List[MethodRecord] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_result(
        self,
        method: str,
        training_result: TrainingResult,
        benchmark_result: BenchmarkResult,
    ) -> None:
        """Register a completed (training, benchmark) pair for one method."""
        self._records.append(
            MethodRecord(
                method=method,
                training=training_result,
                benchmark=benchmark_result,
            )
        )

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def build_table(self) -> pd.DataFrame:
        """Build a side-by-side comparison DataFrame.

        Columns:
            Method, Trainable Params, Total Params, Param Reduction (%),
            Peak Train Memory (MB), Throughput (tok/s), Training Time (s),
            Perplexity, ROUGE-1, ROUGE-2, ROUGE-L,
            Model Size (MB), P50 Latency (ms), Tokens/s (inference)
        """
        rows = []
        for rec in self._records:
            tr = rec.training
            bm = rec.benchmark
            param_reduction = (1 - tr.trainable_fraction) * 100.0
            rows.append(
                {
                    "Method": rec.method,
                    "Trainable Params": tr.trainable_params,
                    "Total Params": tr.total_params,
                    "Param Reduction (%)": round(param_reduction, 2),
                    "Trainable (%)": round(tr.trainable_fraction * 100, 3),
                    "Peak Train Memory (MB)": round(tr.peak_memory_mb, 1),
                    "Throughput (tok/s)": round(tr.mean_throughput_tokens_per_sec, 1),
                    "Training Time (s)": round(tr.training_time_seconds, 1),
                    "Train Loss": round(tr.train_loss, 4),
                    "Eval Loss": round(tr.eval_loss, 4),
                    "Perplexity": round(bm.perplexity, 3),
                    "ROUGE-1": round(bm.rouge1, 4),
                    "ROUGE-2": round(bm.rouge2, 4),
                    "ROUGE-L": round(bm.rougeL, 4),
                    "Model Size (MB)": round(bm.model_size_mb, 1),
                    "P50 Latency (ms)": round(bm.p50_latency_ms, 2),
                    "Tokens/s (inference)": round(bm.tokens_per_second, 1),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted comparison table to stdout.

        Highlights the winning method (lowest is better for memory/latency/
        perplexity; highest for throughput and ROUGE) in each numeric column.
        """
        if not self._records:
            print("No results to display.")
            return

        df = self.build_table()
        print("\n" + "=" * 90)
        print("QLoRA Trainer — Fine-Tuning Method Comparison")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)

        # Determine winners per metric
        lower_better = ["Perplexity", "Eval Loss", "Train Loss", "Peak Train Memory (MB)",
                        "P50 Latency (ms)", "Param Reduction (%)", "Training Time (s)"]
        higher_better = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Throughput (tok/s)",
                         "Tokens/s (inference)"]

        print("\nBest method per metric:")
        for col in lower_better:
            if col in df.columns:
                winner_idx = df[col].idxmin()
                print(f"  {col:<35} -> {df.loc[winner_idx, 'Method']}")
        for col in higher_better:
            if col in df.columns:
                winner_idx = df[col].idxmax()
                print(f"  {col:<35} -> {df.loc[winner_idx, 'Method']}")
        print()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, path: str) -> None:
        """Save comparison as both JSON and CSV at the given base path.

        Writes:
          {path}.json — full nested result tree
          {path}.csv  — flat comparison table
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # CSV
        df = self.build_table()
        df.to_csv(f"{path}.csv", index=False)

        # JSON — combine training + benchmark results
        json_data = []
        for rec in self._records:
            json_data.append(
                {
                    "method": rec.method,
                    "training": asdict(rec.training),
                    "benchmark": asdict(rec.benchmark),
                }
            )
        with open(f"{path}.json", "w") as f:
            json.dump(json_data, f, indent=2, default=_json_serialiser)

        print(f"Report saved to {path}.csv and {path}.json")


def _json_serialiser(obj):
    """Fallback JSON serialiser for non-standard types."""
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
