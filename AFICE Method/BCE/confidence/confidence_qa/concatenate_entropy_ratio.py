import json


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']
index_bias = {'mmlu_0': 0, 'mmlu_1': 1760, 'mmlu_2': 3520, 'mmlu_3': 5280, 'mmlu_4': 7040, 'mmlu_5': 8800, 'mmlu_6': 10560, 'mmlu_7': 12320}
file_to_write = open('mmlu_my_entropy.jsonl', 'w', encoding='utf-8')
root_path = 'vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_q/'
for file_name in file_name_list:
    answers = read_jsonl_to_dicts(root_path + file_name + '_entropy.jsonl')
    for answer  in answers:
        ans_json = {
            "question_id": answer['question_id'] + index_bias[file_name],
            "entropy": answer['entropy'],
        }
        file_to_write.write(json.dumps(ans_json) + "\n")

file_to_write2 = open('mmlu_my_ratio.jsonl', 'w', encoding='utf-8')
root_path2 = 'vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_qa/'
for file_name in file_name_list:
    answers = read_jsonl_to_dicts(root_path2 + file_name + '_ratio.jsonl')
    for answer in answers:
        ans_json = {
            "question_id": answer['question_id'] + index_bias[file_name],
            "ratio": answer['ratio'],
        }
        file_to_write2.write(json.dumps(ans_json) + "\n")
