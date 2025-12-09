import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
import torch

def llama3_api(paragraph,pipeline):
    import prompt

    prompt=prompt.paged_llm_prompt()



    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": paragraph}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=2048,
    )
    return outputs[0]["generated_text"][-1]['content']
