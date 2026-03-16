"""Factory that builds HuggingFace TrainingArguments from our typed TrainConfig."""

from __future__ import annotations

import os

from src.config import TrainConfig

from transformers import TrainingArguments


def get_training_arguments(config: TrainConfig) -> TrainingArguments:
    """Construct a fully-configured TrainingArguments for the requested method.

    Method-specific overrides:
    - full fine-tuning: gradient_checkpointing=True to cap activation memory.
    - qlora: paged_adamw_8bit optimizer when bitsandbytes is available.
    - All: remove_unused_columns=False so that custom datasets with extra
      columns (e.g. 'category') are not silently dropped before the collator.
    """
    output_dir = os.path.join(config.output_dir, config.method)

    # Method-specific defaults that can still be overridden by config.
    gradient_checkpointing = config.method == "full"
    optim = config.optim

    if config.method == "qlora" and optim == "adamw_torch":
        # Prefer paged AdamW for QLoRA: optimizer states spill to CPU RAM
        # during memory pressure spikes instead of causing OOM.
        if _paged_adamw_available():
            optim = "paged_adamw_8bit"

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=optim,
        report_to=config.report_to,
        seed=config.seed,
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if config.max_steps > 0:
        kwargs["max_steps"] = config.max_steps
        # When max_steps overrides epoch count, suppress epoch-based stopping.
        kwargs["num_train_epochs"] = 9999

    return TrainingArguments(**kwargs)


def _paged_adamw_available() -> bool:
    """Return True if bitsandbytes paged optimizer is importable."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False
