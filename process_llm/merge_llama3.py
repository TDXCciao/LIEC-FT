#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge LoRA weights into a full-precision Llama-3-8B-Instruct model (no 4-bit quantization).
Output: float16 full model, ~15 GB
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------------------------------------------------
# Configuration – replace with your own paths or env-vars
# ------------------------------------------------------------------
BASE_MODEL_PATH      = "<BASE_MODEL_PATH>"      # e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
LORA_ADAPTER_PATH    = "<LORA_ADAPTER_PATH>"    # e.g. ./checkpoint-400
MERGED_OUTPUT_PATH   = "<MERGED_OUTPUT_PATH>"   # e.g. ./llama3_8b_logreasoner_full_f16
HF_ENDPOINT          = os.getenv("HF_ENDPOINT", "https://huggingface.co")
HF_TOKEN             = os.getenv("HF_TOKEN",    "<YOUR_HF_TOKEN>")

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_TOKEN"]    = HF_TOKEN


def main():
    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    # ------------------------------------------------------------------
    # 2. Load full-precision base model (float16, no quantization)
    # ------------------------------------------------------------------
    print("Loading full-precision base model (float16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # ------------------------------------------------------------------
    # 3. Load LoRA adapter
    # ------------------------------------------------------------------
    print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # ------------------------------------------------------------------
    # 4. Merge weights
    # ------------------------------------------------------------------
    print("Merging LoRA...")
    model = model.merge_and_unload(progressbar=True)

    # ------------------------------------------------------------------
    # 5. Save merged model
    # ------------------------------------------------------------------
    print(f"Saving full-precision merged model -> {MERGED_OUTPUT_PATH}")
    model.save_pretrained(
        MERGED_OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print("Model merged and saved to:", MERGED_OUTPUT_PATH)


if __name__ == "__main__":
    main()