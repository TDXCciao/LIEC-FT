#!/usr/bin/env python3
"""
QLoRA fine-tuning script for causal language models.
Compatible with Hugging Face Transformers + PEFT.

Usage:
    python train_qlora.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------------------------------------------
# Configuration – replace with your own values or env-vars
# ------------------------------------------------------------------
MODEL_NAME      = "<YOUR_MODEL_NAME>"          # e.g. meta-llama/Llama-2-7b-hf
HF_ENDPOINT     = os.getenv("HF_ENDPOINT", "https://huggingface.co")
HF_TOKEN        = os.getenv("HF_TOKEN", "<YOUR_HF_TOKEN>")
PYTORCH_CUDA_ALLOC_CONF = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TRAIN_DATA_PATH = "<PATH_TO_TRAIN_JSONL>"      # dataset/qlora/train.jsonl
DEV_DATA_PATH   = "<PATH_TO_DEV_JSONL>"        # dataset/qlora/dev.jsonl
OUTPUT_DIR      = "<OUTPUT_DIR>"               # ./llama3_log_stage2

NUM_EPOCHS      = 5
PER_DEVICE_BS   = 1
GRADIENT_ACCUM  = 16
LEARNING_RATE   = 3.0e-5
MAX_SEQ_LEN     = 2048
EVAL_STEPS      = 50
SAVE_STEPS      = 50
LORA_R          = 64
LORA_ALPHA      = 128
LORA_DROPOUT    = 0.05

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_TOKEN"]    = HF_TOKEN
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    legacy=True,
)
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------
# 4-bit quantization config
# ------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ------------------------------------------------------------------
# Model config (handle RoPE scaling if present)
# ------------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    if "rope_type" in config.rope_scaling:
        config.rope_scaling = {
            "type": "dynamic",
            "factor": config.rope_scaling.get("factor", 8.0),
        }

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)

# ------------------------------------------------------------------
# LoRA config
# ------------------------------------------------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ------------------------------------------------------------------
# Dataset preprocessing
# ------------------------------------------------------------------
def format_tokenize(example):
    text = (
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )
    return tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)

data_files = {"train": TRAIN_DATA_PATH, "dev": DEV_DATA_PATH}
raw_datasets = load_dataset("json", data_files=data_files)
train_dataset = raw_datasets["train"].map(
    format_tokenize,
    remove_columns=raw_datasets["train"].column_names,
)
eval_dataset = raw_datasets["dev"].map(
    format_tokenize,
    remove_columns=raw_datasets["dev"].column_names,
)

# ------------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BS,
    per_device_eval_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ------------------------------------------------------------------
# Train & save
# ------------------------------------------------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete. Model saved to {OUTPUT_DIR}")