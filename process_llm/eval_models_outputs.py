# -*- coding: utf-8 -*-
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.empty_cache()
import transformers

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
import llama3_api

# ---------- Configuration ----------
MODEL_NAME = "llama3_no_tuning"
EVALUATION_TYPE = "train"
DATA_PATH = "PATH_TO_YOUR_DATA_FILE"  # Placeholder for data file path
OUTPUT_DIRECTORY = "PATH_TO_OUTPUT_DIRECTORY"  # Placeholder for output directory
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_DIRECTORY, f"{MODEL_NAME}_{EVALUATION_TYPE}.json")
# -----------------------------------

# ---------- Load Model ----------
# pipeline = merge_tokenizer_api.init_model()
pipeline = transformers.pipeline(
    "text-generation",
    model="PATH_TO_YOUR_MODEL",  # Placeholder for model path
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# ---------- Resume from Breakpoint ----------
if os.path.exists(SAVE_PATH):
    save_dict = json.load(open(SAVE_PATH, 'r', encoding='utf-8'))
    completed_set = {item['file_index'] for item in save_dict['model_outputs']}
else:
    save_dict = {"model_name": MODEL_NAME, "eva_type": EVALUATION_TYPE, "model_outputs": []}
    completed_set = set()

# ---------- Data Loading ----------
dataset = json.load(open(DATA_PATH, 'r', encoding='utf-8'))
remaining_data = [d for d in dataset if d['file_index'] not in completed_set]

# ---------- Inference ----------
for data in tqdm(remaining_data, desc="Inferring"):
    print(f"Processing file_index: {data['file_index']}")  # Debug information
    answer = llama3_api.llama3_api(data['paragraph'], pipeline)
    print(f"Got answer for {data['file_index']}: {answer[:50]}...")  # Debug information

    # Preserve all BPMN fields for evaluation
    save_sample = {k: v for k, v in data.items()}  # Deep copy
    save_sample['model_output'] = answer  # Add/update only the model output
    save_dict['model_outputs'].append(save_sample)

    # Real-time saving to disk
    json.dump(save_dict, open(SAVE_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print(f"Completed {len(remaining_data)} items, results saved to {SAVE_PATH}")
