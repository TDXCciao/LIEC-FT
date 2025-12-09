# dpo_train.py
# -*- coding: utf-8 -*-
"""
Script for training a language model using Direct Preference Optimization (DPO) with LoRA.
"""

import torch
import os

# Set environment variables for Hugging Face Hub access
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


################ Configuration ################
MODEL_NAME = "path/to/base/model"
TRAIN_PATH = "path/to/train/data.jsonl"
DEV_PATH = "path/to/dev/data.jsonl"
OUTPUT_DIR = "path/to/output/dir"

NUM_EPOCHS = 5
PER_DEVICE_BS = 1
GRADIENT_ACCUM = 16
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 2048
EVAL_STEPS = 50
SAVE_STEPS = 50
EARLY_STOP_PATIENCE = 5

BETA = 0.1
LOSS_TYPE = "sigmoid"
###############################################


def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Tokenization function
    def tokenize_dpo(example):
        """
        Tokenizes prompt, chosen response, and rejected response.
        Returns tokenized inputs as a dictionary.
        """
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        def _tokenize(text):
            return tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,
            )

        prompt_tokens = _tokenize(prompt)
        chosen_tokens = _tokenize(chosen)
        rejected_tokens = _tokenize(rejected)

        return {
            "prompt": prompt_tokens,
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
        }

    # Load dataset
    raw_datasets = load_dataset("json", data_files={"train": TRAIN_PATH, "dev": DEV_PATH})
    train_data = raw_datasets["train"].map(tokenize_dpo)
    dev_data = raw_datasets["dev"].map(tokenize_dpo)

    # Training arguments
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRADIENT_ACCUM,
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
        beta=BETA,
        loss_type=LOSS_TYPE,
    )

    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        data_collator=None,
    )

    # Start training
    dpo_trainer.train()
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"DPO training completed. Best checkpoint saved at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
