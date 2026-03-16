"""Dolly-15k dataset loading, prompt formatting, and tokenization with label masking."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


RESPONSE_HEADER = "### Response:\n"


def _format_prompt(example: Dict) -> str:
    """Format a Dolly example as an Alpaca-style instruction prompt.

    If the context field is non-empty, it is included under ### Input:.
    Loss is later masked so only the response tokens contribute to training.
    """
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()

    if context:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"{RESPONSE_HEADER}{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"{RESPONSE_HEADER}{response}"
        )
    return text


def _tokenize_and_mask(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict:
    """Tokenize the full prompt and mask instruction tokens in the labels.

    Only tokens after '### Response:\\n' contribute to the loss. Masking
    instruction tokens prevents the model from memorising prompt structure
    instead of learning response generation — a critical detail most
    fine-tuning tutorials skip.
    """
    full_text = _format_prompt(example)

    tokenized = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids: List[int] = tokenized["input_ids"]
    labels: List[int] = list(input_ids)

    # Encode the response header alone (without special tokens so we can
    # locate the exact token boundary).
    header_ids: List[int] = tokenizer.encode(
        RESPONSE_HEADER, add_special_tokens=False
    )
    header_len = len(header_ids)

    # Slide a window over input_ids to find the first occurrence of the
    # response header token sequence.
    response_start = -1
    for i in range(len(input_ids) - header_len + 1):
        if input_ids[i : i + header_len] == header_ids:
            response_start = i + header_len  # first token of the actual response
            break

    if response_start == -1:
        # Header was truncated away; mask everything so the example is skipped
        # during loss computation.
        labels = [-100] * len(input_ids)
    else:
        for i in range(response_start):
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


class DollyDataset:
    """Manages loading, splitting, and tokenizing the Dolly-15k dataset."""

    DATASET_ID = "databricks/databricks-dolly-15k"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_fraction = val_fraction
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw(self) -> Dataset:
        """Download (or load from cache) the full Dolly-15k training split."""
        ds: DatasetDict = load_dataset(self.DATASET_ID)
        return ds["train"]

    @staticmethod
    def format_prompt(example: Dict) -> str:
        """Public wrapper so callers can access prompt formatting directly."""
        return _format_prompt(example)

    def get_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        """Return tokenized train and validation datasets.

        The validation split is created by stratified sampling on the
        'category' field so that all eight Dolly task types are represented
        proportionally in both splits.
        """
        raw = self.load_raw()
        raw = raw.shuffle(seed=self.seed)

        # Stratified split by category
        categories: List[str] = list(set(raw["category"]))
        train_indices: List[int] = []
        val_indices: List[int] = []

        for cat in categories:
            cat_indices = [i for i, c in enumerate(raw["category"]) if c == cat]
            n_val = max(1, int(len(cat_indices) * self.val_fraction))
            val_indices.extend(cat_indices[:n_val])
            train_indices.extend(cat_indices[n_val:])

        train_raw = raw.select(train_indices)
        val_raw = raw.select(val_indices)

        train_ds = self._tokenize_dataset(train_raw)
        val_ds = self._tokenize_dataset(val_raw)

        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        tokenized = dataset.map(
            lambda ex: _tokenize_and_mask(ex, self.tokenizer, self.max_length),
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        # Drop examples where the entire label sequence was masked (truncation)
        tokenized = tokenized.filter(
            lambda ex: any(l != -100 for l in ex["labels"]),
            desc="Filtering masked-only examples",
        )
        return tokenized
