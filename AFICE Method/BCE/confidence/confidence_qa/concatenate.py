import json


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']
index_bias = {'mmlu_0': 0, 'mmlu_1': 1760, 'mmlu_2': 3520, 'mmlu_3': 5280, 'mmlu_4': 7040, 'mmlu_5': 8800, 'mmlu_6': 10560, 'mmlu_7': 12320}
file_to_write = open('mmlu_train_question_answer_my_method.jsonl', 'w', encoding='utf-8')
for file_name in file_name_list:
    answers = read_jsonl_to_dicts('vicuna/finetune_method/my_method/generate_train_data/train_question/model_answer/' + file_name + '_train_question_greedy_answer.jsonl')
    confidences = read_jsonl_to_dicts('vicuna/finetune_method/my_method/generate_train_data/confidence/final_confidence/' + file_name + '_confidence_qa.jsonl')
    answers_dict = {item['question_id']:item['choices'][0]['turns'][0] for item in answers}
    confidences_dict = {item['question_id']:item['confidence'] for item in confidences}
    intersection_id_list = list(set(answers_dict.keys()) & set(confidences_dict.keys()))
    for id in intersection_id_list:
        ans_json = {
            "question_id": id + index_bias[file_name],
            "answer": answers_dict[id],
            "confidence": confidences_dict[id],
        }
        file_to_write.write(json.dumps(ans_json) + "\n")
