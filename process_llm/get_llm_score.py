#!/usr/bin/env python3
"""
Evaluate generated outputs against reference answers using an OpenAI-compatible API.
Results are appended to a JSONL file for downstream analysis.
"""

import os
import json
from openai import OpenAI

# ------------------------------------------------------------------
# Configuration ¨C replace with your own values or environment variables
# ------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL",   "<YOUR_OPENAI_MODEL>")

REFERENCE_PATH = "<PATH_TO_REFERENCE_JSONL>"          # e.g. dataset/qlora/test.jsonl
GENERATED_PATH = "<PATH_TO_GENERATED_JSON>"           # e.g. llama3_8b_logreasoner_merged_test.json
OUTPUT_PATH    = "<PATH_TO_OUTPUT_JSONL>"             # e.g. gpt_score/llama3_8b_logreasoner_merged_test.jsonl

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------------
# Helper: call OpenAI API
# ------------------------------------------------------------------
def openai_api(user_prompt: str) -> str:
    """Return the stripped text from the OpenAI response."""
    response = client.responses.create(model=OPENAI_MODEL, input=user_prompt)
    return response.output_text.strip()


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main() -> None:
    # Load reference samples
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        ref_samples = [json.loads(line) for line in f]

    # Load generated outputs
    with open(GENERATED_PATH, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    gen_samples = gen_data.get("model_outputs", [])

    assert len(ref_samples) == len(gen_samples), "Mismatch between reference and generated counts!"

    for idx, (ref, gen) in enumerate(zip(ref_samples, gen_samples)):
        print(f"Processing {idx}")
        prompt_text = (
            f"correct reference sample: {ref['output']}\n"
            f"generated sample: {gen['model_output']}"
        )
        score = openai_api(prompt_text)

        record = {"prompt": ref["input"], "score": score}
        print(record)

        # Append to output JSONL
        with open(OUTPUT_PATH, "a", encoding="utf-8") as outf:
            json.dump(record, outf, ensure_ascii=False)
            outf.write("\n")

    print(f"Finished. Total evaluated: {len(ref_samples)}")


if __name__ == "__main__":
    main()