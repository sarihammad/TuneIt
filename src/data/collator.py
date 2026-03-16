"""Data collator for causal LM fine-tuning with dynamic padding."""

from __future__ import annotations

from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase


def build_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> DataCollatorForSeq2Seq:
    """Return a DataCollatorForSeq2Seq configured for causal LM fine-tuning.

    Padding is aligned to multiples of 8 for tensor-core efficiency on
    Ampere/Ada GPUs. Labels are padded with -100 so that padding positions
    are automatically ignored by CrossEntropyLoss.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
