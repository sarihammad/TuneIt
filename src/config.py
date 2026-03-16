"""Central configuration using Pydantic models for type-safe, validated settings."""

from __future__ import annotations

from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    model_name: str = "facebook/opt-125m"
    max_length: int = 512
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "float32"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"

    @field_validator("torch_dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        valid = {"float16", "bfloat16", "float32"}
        if v not in valid:
            raise ValueError(f"torch_dtype must be one of {valid}, got {v!r}")
        return v


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @field_validator("r")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {v}")
        return v


class QLoRAConfig(BaseModel):
    r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, v: int) -> int:
        if v not in {4, 8}:
            raise ValueError(f"bits must be 4 or 8, got {v}")
        return v


class TrainConfig(BaseModel):
    method: str = "lora"  # "full", "lora", "qlora"
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    report_to: str = "none"
    seed: int = 42
    max_steps: int = -1  # -1 means use num_epochs

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {"full", "lora", "qlora"}
        if v not in valid:
            raise ValueError(f"method must be one of {valid}, got {v!r}")
        return v


class AppConfig(BaseModel):
    """Top-level config combining all sub-configs, loadable from YAML."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: Optional[LoRAConfig] = None
    qlora: Optional[QLoRAConfig] = None
    training: TrainConfig = Field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def get_lora_config(self) -> LoRAConfig:
        """Return LoRAConfig, falling back to defaults if not specified."""
        if self.lora is not None:
            return self.lora
        return LoRAConfig()

    def get_qlora_config(self) -> QLoRAConfig:
        """Return QLoRAConfig, falling back to defaults if not specified."""
        if self.qlora is not None:
            return self.qlora
        return QLoRAConfig()
