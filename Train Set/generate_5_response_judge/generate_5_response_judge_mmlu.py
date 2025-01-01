import json
import re


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


file_to_write = open('mmlu_beam_sample_question_answer_judge.jsonl', 'w')
true_answers = read_jsonl_to_dicts('../ture_answer/mmlu_beam_sample_question_answer.jsonl')
datas = read_jsonl_to_dicts('../mmlu.jsonl')
for answer in true_answers:
    if answer['choices'][0]['turns'][0] == 'ERROR':
        continue
    model_answer = answer['choices'][0]['turns'][0]
    id = int(answer['question_id'])
    correct_answer = '(' + datas[id]['answer'] + ')'
    a, b, c, d = 0, 0, 0, 0
    if '(A)' in model_answer:
        a += 1
    if '(B)' in model_answer:
        b += 1
    if '(C)' in model_answer:
        c += 1
    if '(D)' in model_answer:
        d += 1
    if a == 0 and b == 0 and c == 0 and d == 0:
        tag = 'wrong'
    elif a + b + c + d > 1:
        tag = 'wrong'
    else:
        if correct_answer in model_answer:
            tag = 'correct'
        else:
            tag = 'wrong'
    dict_to_write = {}
    dict_to_write['question_id'] = id
    dict_to_write['tag'] = tag
    final_json_str = json.dumps(dict_to_write)
    file_to_write.write(final_json_str + '\n')
