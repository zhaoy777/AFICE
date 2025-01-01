import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_question_and_answer(question_path, answer_path):
    sequences = []
    answers = read_jsonl_to_dicts(answer_path)

    for answer in answers:
        id = answer['question_id']
        question_tmp = ""
        answer_list = answer['choices']
        answer_list = [dict_item['turns'][0] for dict_item in answer_list]
        sequences.append({'id': id, 'question': question_tmp, 'answer_list': answer_list})

    return sequences

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

file_list = ['mmlu']


for category_name in file_list:
    question_path = category_name + '_train_question.jsonl'
    answer_path = 'vicuna/finetune_method/my_method/generate_train_data/train_question/model_answer/' + category_name + '_train_question_5_answer.jsonl'
    file_to_write = open(category_name + '_semantic_entropy_answer.jsonl', 'w')
    sequences = load_question_and_answer(question_path, answer_path)

    for sample in tqdm(sequences):
        question = sample['question']
        answer_list = sample['answer_list']

        semantic_set_ids = {}
        for index, answer in enumerate(answer_list):
            semantic_set_ids[answer] = index


        try:
            if len(answer_list) > 1:
                # Evalauate semantic similarity
                for i, reference_answer in enumerate(answer_list):
                    for j in range(i + 1, len(answer_list)):

                        if semantic_set_ids[answer_list[i]] == i and semantic_set_ids[answer_list[j]] == j:

                            qa_1 = question + ' ' + answer_list[i]
                            qa_2 = question + ' ' + answer_list[j]

                            input = qa_1 + ' [SEP] ' + qa_2
                            encoded_input = tokenizer.encode(input, padding=True)
                            prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                            predicted_label = torch.argmax(prediction, dim=1)

                            reverse_input = qa_2 + ' [SEP] ' + qa_1
                            encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                            reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                            reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                            if 0 in predicted_label or 0 in reverse_predicted_label:
                                has_semantically_different_answers = True
                            else:
                                semantic_set_ids[answer_list[j]] = semantic_set_ids[answer_list[i]]
        except Exception as e:
            print('id', sample['id'])
            continue

        index_set = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        value_list = semantic_set_ids.values()
        for index in value_list:
            index_set[index] += 1
        entropy = 0
        cluster_total_num = 0
        for index, value in index_set.items():
            if value != 0:
                cluster_total_num += 1
                entropy += math.log(value / 5, math.e)
        semantic_entropy = (-1) * entropy / cluster_total_num

        dict_to_write_llm_false_partially_disagree = {}
        dict_to_write_llm_false_partially_disagree['question_id'] = sample['id']
        dict_to_write_llm_false_partially_disagree['semantic_entropy'] = semantic_entropy
        final_json_str = json.dumps(dict_to_write_llm_false_partially_disagree)
        file_to_write.write(final_json_str + '\n')


