"""Benchmark pipeline: train all three methods and produce a comparison report."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train full/LoRA/QLoRA and compare results"
    )
    p.add_argument(
        "--model-name",
        default="facebook/opt-125m",
        help="Base model to benchmark",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Steps per method (default 100 for quick benchmark)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for checkpoints",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory for comparison reports",
    )
    p.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full fine-tuning (fastest to run lora + qlora only)",
    )
    p.add_argument(
        "--n-perplexity",
        type=int,
        default=100,
        help="Eval examples for perplexity benchmark",
    )
    p.add_argument(
        "--n-rouge",
        type=int,
        default=50,
        help="Eval examples for ROUGE benchmark",
    )
    p.add_argument(
        "--n-latency",
        type=int,
        default=20,
        help="Inference runs for latency benchmark",
    )
    return p.parse_args()


def run_method(method: str, model_name: str, max_steps: int, output_dir: str):
    """Train one method and return (TrainingResult, model, tokenizer, eval_dataset)."""
    from src.config import AppConfig, LoRAConfig, ModelConfig, QLoRAConfig, TrainConfig
    from src.training.trainer import MethodTrainer

    model_cfg = ModelConfig(model_name=model_name, torch_dtype="float32")
    train_cfg = TrainConfig(
        method=method,
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_steps=max_steps,
        save_steps=max_steps,
        report_to="none",
    )

    adapter_cfg = None
    if method == "lora":
        adapter_cfg = LoRAConfig()
    elif method == "qlora":
        adapter_cfg = QLoRAConfig()
        model_cfg.load_in_4bit = True

    trainer = MethodTrainer(
        method=method,
        model_config=model_cfg,
        train_config=train_cfg,
        adapter_config=adapter_cfg,
    )
    trainer.setup()
    result = trainer.train()

    return result, trainer.model, trainer.tokenizer, trainer.eval_dataset


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    from src.evaluation.benchmark import BenchmarkEvaluator
    from src.evaluation.comparison import ComparisonBuilder

    builder = ComparisonBuilder()
    methods = ["lora", "qlora"] if args.skip_full else ["full", "lora", "qlora"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Training: {method.upper()}")
        print(f"{'='*60}")

        try:
            training_result, model, tokenizer, eval_dataset = run_method(
                method=method,
                model_name=args.model_name,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"ERROR training {method}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"\nRunning benchmarks for {method}...")
        evaluator = BenchmarkEvaluator(
            model=model,
            tokenizer=tokenizer,
            method=method,
        )

        try:
            benchmark_result = evaluator.run_all(
                eval_dataset=eval_dataset,
                n_perplexity_samples=args.n_perplexity,
                n_rouge_samples=args.n_rouge,
                n_latency_runs=args.n_latency,
            )
        except Exception as e:
            print(f"ERROR benchmarking {method}: {e}")
            import traceback
            traceback.print_exc()
            continue

        builder.add_result(
            method=method,
            training_result=training_result,
            benchmark_result=benchmark_result,
        )

        # Free GPU memory between methods
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not builder._records:
        print("\nNo results collected. Check errors above.")
        sys.exit(1)

    # Print and save
    builder.print_report()
    report_path = os.path.join(args.results_dir, f"comparison_{timestamp}")
    builder.save_report(report_path)

    print(f"\nBenchmark complete. Reports saved to:")
    print(f"  {report_path}.csv")
    print(f"  {report_path}.json")


if __name__ == "__main__":
    main()
