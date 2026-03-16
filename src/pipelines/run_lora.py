"""LoRA fine-tuning pipeline: low-rank adapter training in full precision."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning on Dolly-15k")
    p.add_argument("--config", default="configs/lora.yaml", help="Path to YAML config")
    p.add_argument("--model-name", default=None, help="Override model_name in config")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--output-dir", default=None, help="Override output_dir")
    p.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.config import AppConfig, LoRAConfig
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

    cfg.training.method = "lora"
    lora_cfg = cfg.get_lora_config()
    if args.lora_r is not None:
        lora_cfg = LoRAConfig(
            r=args.lora_r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
        )

    print(f"Starting LoRA fine-tuning: {cfg.model.model_name}")
    print(f"  r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
    print(f"  target_modules={lora_cfg.target_modules}")
    print(f"  max_steps={cfg.training.max_steps}, epochs={cfg.training.num_train_epochs}")

    trainer = MethodTrainer(
        method="lora",
        model_config=cfg.model,
        train_config=cfg.training,
        adapter_config=lora_cfg,
    )
    trainer.setup()
    result = trainer.train()

    print("\n=== LoRA Fine-Tuning Complete ===")
    print(f"  Eval loss:        {result.eval_loss:.4f}")
    print(f"  Perplexity:       {result.perplexity:.3f}")
    print(f"  Training time:    {result.training_time_seconds:.1f}s")
    print(f"  Peak memory:      {result.peak_memory_mb:.1f} MB")
    print(f"  Trainable params: {result.trainable_params:,} / {result.total_params:,} "
          f"({result.trainable_fraction * 100:.2f}%)")
    print(f"  Checkpoint:       {result.checkpoint_path}")


if __name__ == "__main__":
    main()
