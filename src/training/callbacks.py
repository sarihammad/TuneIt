"""Custom HuggingFace Trainer callbacks for memory tracking and throughput logging."""

from __future__ import annotations

import time
from typing import List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MemoryTrackingCallback(TrainerCallback):
    """Records peak GPU memory allocation at each optimiser step.

    Peak memory is measured via torch.cuda.max_memory_allocated() which
    returns the all-time high since the last reset, making it robust to
    transient spikes. On CPU-only machines the callback is a no-op.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.memory_samples: List[float] = []
        self._has_cuda = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # TrainerCallback interface
    # ------------------------------------------------------------------

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._has_cuda:
            return
        if state.global_step % self.log_every_n_steps == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.memory_samples.append(peak_mb)
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._has_cuda:
            return
        # Capture any remaining memory after the final step.
        final_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        if final_peak_mb > 0:
            self.memory_samples.append(final_peak_mb)
        torch.cuda.reset_peak_memory_stats()

        if self.memory_samples:
            print(
                f"\n[MemoryCallback] Peak GPU memory — "
                f"max: {max(self.memory_samples):.1f} MB, "
                f"mean: {sum(self.memory_samples) / len(self.memory_samples):.1f} MB"
            )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def peak_memory_mb(self) -> float:
        """Return the maximum recorded peak memory in MB."""
        return max(self.memory_samples) if self.memory_samples else 0.0

    def mean_memory_mb(self) -> float:
        """Return the mean peak memory across all recorded steps."""
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples)


class ThroughputCallback(TrainerCallback):
    """Measures training throughput in tokens per second.

    Throughput = (batch_size * gradient_accumulation_steps * seq_len) / wall_time
    This gives a hardware-normalised measure of training efficiency that is
    directly comparable across different GPU SKUs and sequence lengths.
    """

    def __init__(self) -> None:
        self.throughput_samples: List[float] = []
        self._step_start_time: Optional[float] = None
        self._batch_tokens: Optional[int] = None

    # ------------------------------------------------------------------
    # TrainerCallback interface
    # ------------------------------------------------------------------

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._step_start_time = time.perf_counter()
        # Best-effort token count; may not be available at step_begin.
        self._batch_tokens = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * _get_world_size()
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self._step_start_time is None:
            return
        elapsed = time.perf_counter() - self._step_start_time
        if elapsed > 0 and self._batch_tokens is not None:
            # Use max_length as a conservative token count per sample.
            tokens_in_step = self._batch_tokens * _get_max_length(kwargs)
            throughput = tokens_in_step / elapsed
            self.throughput_samples.append(throughput)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.throughput_samples:
            mean_tp = sum(self.throughput_samples) / len(self.throughput_samples)
            print(
                f"\n[ThroughputCallback] Training throughput — "
                f"mean: {mean_tp:.0f} tokens/s"
            )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def mean_throughput(self) -> float:
        if not self.throughput_samples:
            return 0.0
        return sum(self.throughput_samples) / len(self.throughput_samples)

    def peak_throughput(self) -> float:
        return max(self.throughput_samples) if self.throughput_samples else 0.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _get_max_length(kwargs: dict) -> int:
    """Try to infer the sequence length from the model config in kwargs."""
    model = kwargs.get("model")
    if model is not None and hasattr(model, "config"):
        cfg = model.config
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return min(int(val), 512)
    return 512
