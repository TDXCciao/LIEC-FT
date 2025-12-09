import json

datasets_path = 'evaluation/gpt_score/'
with open(datasets_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [json.loads(line)['score'] for line in lines]
print(lines[0])
Faithfulness = 0
Correctness = 0
Complexity = 0
Readability = 0
Start_End = 0
for score in lines:
    score = json.loads(score)
    Faithfulness+=score["Faithfulness"]
    Correctness+=score["Correctness"]
    Complexity+=score["Complexity"]
    Readability+=score["Readability"]
    Start_End+=score["Start/End"]
print("Faithfulness:", Faithfulness/len(lines), "Correctness:", Correctness/len(lines), "Complexity:", Complexity/len(lines), "Readability:", Readability/len(lines), "Start/End:", Start_End/len(lines))