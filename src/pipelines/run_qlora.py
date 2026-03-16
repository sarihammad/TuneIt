"""QLoRA fine-tuning pipeline: 4-bit NF4 quantisation + LoRA adapters."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning on Dolly-15k")
    p.add_argument("--config", default="configs/qlora.yaml", help="Path to YAML config")
    p.add_argument("--model-name", default=None, help="Override model_name in config")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--output-dir", default=None, help="Override output_dir")
    p.add_argument("--lora-r", type=int, default=None, help="Override QLoRA rank")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.config import AppConfig, QLoRAConfig
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

    cfg.training.method = "qlora"
    cfg.model.load_in_4bit = True

    qlora_cfg = cfg.get_qlora_config()
    if args.lora_r is not None:
        qlora_cfg = QLoRAConfig(
            r=args.lora_r,
            lora_alpha=qlora_cfg.lora_alpha,
            target_modules=qlora_cfg.target_modules,
            lora_dropout=qlora_cfg.lora_dropout,
            bias=qlora_cfg.bias,
            bits=qlora_cfg.bits,
            double_quant=qlora_cfg.double_quant,
            quant_type=qlora_cfg.quant_type,
        )

    print(f"Starting QLoRA fine-tuning: {cfg.model.model_name}")
    print(f"  r={qlora_cfg.r}, alpha={qlora_cfg.lora_alpha}, bits={qlora_cfg.bits}")
    print(f"  quant_type={qlora_cfg.quant_type}, double_quant={qlora_cfg.double_quant}")
    print(f"  target_modules={qlora_cfg.target_modules}")
    print(f"  max_steps={cfg.training.max_steps}, epochs={cfg.training.num_train_epochs}")

    trainer = MethodTrainer(
        method="qlora",
        model_config=cfg.model,
        train_config=cfg.training,
        adapter_config=qlora_cfg,
    )
    trainer.setup()
    result = trainer.train()

    print("\n=== QLoRA Fine-Tuning Complete ===")
    print(f"  Eval loss:        {result.eval_loss:.4f}")
    print(f"  Perplexity:       {result.perplexity:.3f}")
    print(f"  Training time:    {result.training_time_seconds:.1f}s")
    print(f"  Peak memory:      {result.peak_memory_mb:.1f} MB")
    print(f"  Trainable params: {result.trainable_params:,} / {result.total_params:,} "
          f"({result.trainable_fraction * 100:.2f}%)")
    print(f"  Checkpoint:       {result.checkpoint_path}")


if __name__ == "__main__":
    main()
