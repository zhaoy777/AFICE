import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_file_probability():
    probability_dict = {'mmlu_0': {}, 'mmlu_1': {}, 'mmlu_2': {}, 'mmlu_3': {}, 'mmlu_4': {}, 'mmlu_5': {}, 'mmlu_6': {}, 'mmlu_7': {}}

    root_path = 'vicuna/finetune_method/my_method/generate_train_data/train_question/model_answer/'
    file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']
    for file_name in file_name_list:
        print('file_name:', file_name)
        datas = read_jsonl_to_dicts(root_path + file_name + '_train_question_beam_sample_20_answer.jsonl')
        for content in datas:
            try:
                vector = [item / len(content['scores_list']) for item in content['scores_list'][-1]]
            except Exception as e:
                # print(content['question_id'], end='@ ')
                continue
            # vector = torch.tensor(vector)
            probability_dict[file_name][content['question_id']] = vector
    return probability_dict

file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']

probability_dict = load_file_probability()
for file_name in file_name_list:
    file_to_write = open('vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_qa/' + file_name + '_ratio.jsonl', 'w')
    id_list = probability_dict[file_name].keys()
    file_path = 'vicuna/finetune_method/my_method/generate_train_data/train_question/model_answer/' + file_name + '_train_question_greedy_answer.jsonl'
    datas = read_jsonl_to_dicts(file_path)
    for data in datas:
        if data['question_id'] in id_list:
            scores_list = [math.log(item[0]) for item in data['scores_list']]
            normalize_log_score = sum(scores_list) / len(scores_list)
            if data['question_id'] % 100 == 0:
                print('normalize_log_score', normalize_log_score)
                print(probability_dict[file_name][data['question_id']])
            culmulative_probability = normalize_log_score
            for i in range(len(probability_dict[file_name][data['question_id']])):
                if probability_dict[file_name][data['question_id']][i] <= normalize_log_score:
                    culmulative_probability += probability_dict[file_name][data['question_id']][i]
            ratio = culmulative_probability / (sum(probability_dict[file_name][data['question_id']]) + normalize_log_score)
            ans_json = {
                "question_id": data['question_id'],
                "ratio": float(ratio)
            }
            file_to_write.write(json.dumps(ans_json) + "\n")
