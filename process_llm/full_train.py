#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-parameter fine-tuning Llama-3.1-8B-Instruct with DeepSpeed ZeRO-3
"""

import os
import json
import torch
import warnings
warnings.filterwarnings("ignore")

# Environment variables
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"  # Replace with your Hugging Face token
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from trl import DataCollatorForCompletionOnlyLM

################################ Hyperparameters ################################
MODEL_NAME = "PATH_TO_MODEL"  # Placeholder for model path
TRAIN_DATA_PATH = "PATH_TO_TRAIN_DATA"  # Placeholder for training data path
DEV_DATA_PATH = "PATH_TO_DEV_DATA"  # Placeholder for development data path
OUTPUT_DIR = "PATH_TO_OUTPUT_DIR"  # Placeholder for output directory
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 1024
EVAL_STEPS = 50
SAVE_STEPS = 50
EARLY_STOPPING_PATIENCE = 5
#################################################################################

#################### 1. Tokenizer + Chat Template ####################
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

LLAMA3_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = '<|begin_of_text|>' + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
tokenizer.chat_template = LLAMA3_TEMPLATE

########################### 2. Dataset Preparation ##############################
def format_and_tokenize(example):
    """
    Format the conversation and tokenize it
    """
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    tokens = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH
    )
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

# Load datasets
raw_datasets = load_dataset(
    "json", 
    data_files={
        "train": TRAIN_DATA_PATH, 
        "dev": DEV_DATA_PATH
    }, 
    split={
        "train": "train", 
        "dev": "dev"
    }
)

train_dataset = raw_datasets["train"].map(
    format_and_tokenize, 
    remove_columns=raw_datasets["train"].column_names
)
dev_dataset = raw_datasets["dev"].map(
    format_and_tokenize, 
    remove_columns=raw_datasets["dev"].column_names
)

########################### 3. Model Initialization #############################
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

########################### 4. Training Arguments ###############################
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
    deepspeed="ds_zero3.json",  # Key configuration for DeepSpeed
)

########################### 5. Data Collator ####################################
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
    mlm=False
)

########################### 6. Trainer Setup ####################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

########################### 7. Training & Saving ################################
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(" Full-parameter fine-tuning completed. Best checkpoint saved to:", OUTPUT_DIR)
