"""Full fine-tuning pipeline: all parameters updated, no adapters."""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full fine-tuning on Dolly-15k")
    p.add_argument("--config", default="configs/full_finetune.yaml", help="Path to YAML config")
    p.add_argument("--model-name", default=None, help="Override model_name in config")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--output-dir", default=None, help="Override output_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.config import AppConfig, ModelConfig, TrainConfig
    from src.training.trainer import MethodTrainer

    if os.path.exists(args.config):
        cfg = AppConfig.from_yaml(args.config)
    else:
        print(f"Config not found at {args.config}, using defaults.")
        cfg = AppConfig()

    if args.model_name:
        cfg.model.model_name = args.model_name
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    cfg.training.method = "full"

    print(f"Starting full fine-tuning: {cfg.model.model_name}")
    print(f"  max_steps={cfg.training.max_steps}, epochs={cfg.training.num_train_epochs}")
    print(f"  batch_size={cfg.training.per_device_train_batch_size}, "
          f"grad_accum={cfg.training.gradient_accumulation_steps}")

    trainer = MethodTrainer(
        method="full",
        model_config=cfg.model,
        train_config=cfg.training,
    )
    trainer.setup()
    result = trainer.train()

    print("\n=== Full Fine-Tuning Complete ===")
    print(f"  Eval loss:        {result.eval_loss:.4f}")
    print(f"  Perplexity:       {result.perplexity:.3f}")
    print(f"  Training time:    {result.training_time_seconds:.1f}s")
    print(f"  Peak memory:      {result.peak_memory_mb:.1f} MB")
    print(f"  Trainable params: {result.trainable_params:,} / {result.total_params:,}")
    print(f"  Checkpoint:       {result.checkpoint_path}")


if __name__ == "__main__":
    main()
