# -*- coding: utf-8 -*-
import json


def convert_dev_to_jsonl(input_file, output_file):
    """
    将dev.json格式转换为JSONL格式
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if line:
                try:
                    # 解析每一行的JSON
                    item = json.loads(line)

                    # 构建messages格式
                    # messages_format = {
                    #     "input": item['paragraph'],
                    #     "output": item['answer']
                    # }
                    messages_format = {
                        "prompt": item['prompt'],
                        "chosen": item['chosen'],
                        "rejected": item['rejected']
                    }

                    # 写入JSONL文件
                    f.write(json.dumps(messages_format, ensure_ascii=False) + '\n')

                except json.JSONDecodeError as e:
                    print(f"解析错误在行: {line[:50]}... 错误: {e}")


# 执行转换
convert_dev_to_jsonl('dpo_datasets/no_lora_dpo_train.json', 'dpo_jsonl/llama3_no_tuning_train.jsonl')
print("转换完成！")
