import json
import math


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']
for file_name in file_name_list:
    entropys = read_jsonl_to_dicts('vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_q/' + file_name + '_entropy.jsonl')
    ratios = read_jsonl_to_dicts('vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_qa/' + file_name + '_ratio.jsonl')
    entropy_dict = {item['question_id']:item['entropy'] for item in entropys}
    ratio_dict = {item['question_id']:item['ratio'] for item in ratios}
    intersection_id_list = list(set(entropy_dict.keys()) & set(ratio_dict.keys()))
    file_to_write = open(
        'vicuna/finetune_method/my_method/generate_train_data/confidence/final_confidence/' + file_name + '_confidence_qa.jsonl', 'w')
    for id in intersection_id_list:
        ans_json = {
            "question_id": id,
            "confidence": float(math.pow(ratio_dict[id], 0.2) * math.exp(-entropy_dict[id] * 1.1)),
        }
        file_to_write.write(json.dumps(ans_json) + "\n")
