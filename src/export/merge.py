"""Merge LoRA adapters into the base model and export to disk."""

from __future__ import annotations

import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


class AdapterMerger:
    """Handles merging LoRA/QLoRA adapters into the base model weights.

    After merging, the model behaves as a standard (non-PEFT) model with
    no inference overhead from the adapter. The merged checkpoint can be
    deployed on any serving stack that accepts HuggingFace model format.
    """

    @staticmethod
    def merge_lora_into_base(
        peft_model: PeftModel,
        output_dir: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> str:
        """Merge adapter delta weights (B @ A) into the frozen base weights.

        merge_and_unload() computes W_merged = W_base + (lora_alpha/r) * B @ A
        for every adapted layer, producing a standard dense model. The result
        is identical to a full fine-tune of those layers but was trained at a
        fraction of the memory cost.

        Args:
            peft_model: Trained PEFT model with LoRA/QLoRA adapters.
            output_dir: Directory to save merged model + tokenizer.
            tokenizer: If provided, saved alongside the model.

        Returns:
            Absolute path to the saved directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Merging LoRA adapters into base model weights...")
        merged_model = peft_model.merge_and_unload()

        print(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(output_dir, safe_serialization=True)

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        print(f"Merge complete. Checkpoint saved to {os.path.abspath(output_dir)}")
        return os.path.abspath(output_dir)

    @staticmethod
    def export_gguf_instructions(model_dir: str) -> str:
        """Return step-by-step instructions for exporting to GGUF format.

        GGUF is the binary format used by llama.cpp and ollama for
        quantised CPU inference. These steps assume the merged model
        (produced by merge_lora_into_base) is already saved to disk.
        """
        instructions = f"""
=== Exporting to GGUF (llama.cpp) ===

Prerequisites:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make -j$(nproc)
  pip install -r requirements.txt

Step 1 — Convert HuggingFace checkpoint to GGUF (F16):
  python convert_hf_to_gguf.py \\
      {model_dir} \\
      --outfile merged_model_f16.gguf \\
      --outtype f16

Step 2 — Quantise to Q4_K_M (4-bit, recommended quality/size tradeoff):
  ./llama-quantize merged_model_f16.gguf merged_model_q4_k_m.gguf Q4_K_M

Step 3 — Test inference:
  ./llama-cli \\
      -m merged_model_q4_k_m.gguf \\
      -p "### Instruction:\\nExplain gradient descent.\\n\\n### Response:\\n" \\
      --n-predict 200

Step 4 — Serve with ollama (optional):
  ollama create my-model -f ./Modelfile
  ollama run my-model

Note: Q4_K_M offers the best balance of size and perplexity degradation.
      Q5_K_M retains more quality at 25% more disk space.
      Q8_0 is near-lossless at 2x the size of Q4_K_M.
"""
        print(instructions)
        return instructions

    @staticmethod
    def quantize_to_8bit(
        model_dir: str,
        tokenizer_dir: str,
        output_dir: str,
    ) -> str:
        """Reload a merged model in 8-bit precision and save to output_dir.

        This uses bitsandbytes INT8 linear quantisation (LLM.int8()) which
        applies mixed-precision quantisation: outlier features are kept in
        FP16 while the majority of weights are quantised to INT8, achieving
        roughly 2x memory savings with negligible quality loss.

        Args:
            model_dir: Path to the merged (non-quantised) HuggingFace model.
            tokenizer_dir: Path to the tokenizer directory.
            output_dir: Directory where the 8-bit model will be saved.

        Returns:
            Absolute path to the saved directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading {model_dir} in 8-bit precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            load_in_8bit=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        print(f"Saving 8-bit model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"8-bit export complete: {os.path.abspath(output_dir)}")
        return os.path.abspath(output_dir)
