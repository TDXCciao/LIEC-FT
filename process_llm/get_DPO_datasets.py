import os
import json
from openai import OpenAI

# ------------------------------------------------------------------
# Configuration – replace with your own values or environment variables
# ------------------------------------------------------------------
OPENAI_API_KEY      = "<YOUR_OPENAI_API_KEY>"
OPENAI_MODEL        = "<YOUR_OPENAI_MODEL>"
DATASET_PATH        = "<PATH_TO_TRAIN_JSONL>"
DPO_PATH            = "<PATH_TO_DPO_JSON>"
OUTPUT_PATH         = "<PATH_TO_OUTPUT_JSONL>"

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------------
# Helper: call OpenAI API
# ------------------------------------------------------------------
def openai_api(user_prompt):

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=user_prompt
    )
    return response.output_text.strip()


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main():
    # Load training examples
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        train_samples = [json.loads(line) for line in f]

    # Load DPO-style model outputs
    with open(DPO_PATH, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)
    dpo_outputs = dpo_data.get("model_outputs", [])

    assert len(train_samples) == len(dpo_outputs), "Mismatched lengths!"

    # Process each pair
    for idx, (sample, dpo) in enumerate(zip(train_samples, dpo_outputs)):
        print(f"Processing sample {idx}")
        prompt_text = (
            f"PARAGRAPH: {sample['output']}\n"
            f"REJECTED: {dpo['model_output']}"
        )
        answer = openai_api(prompt_text)

        if answer.lower() != "correct":
            record = {
                "prompt": sample["input"],
                "answer": sample["output"],
                "chosen": answer,
                "rejected": dpo["model_output"]
            }
            # Append to output file
            with open(OUTPUT_PATH, "a", encoding="utf-8") as outf:
                json.dump(record, outf, ensure_ascii=False)
                outf.write("\n")

    print(f"Finished. Total samples processed: {len(train_samples)}")


if __name__ == "__main__":
    main()