import json


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

prompt3 = ("I will give you a multiple-choice question with four options and the correct answer. Please explain why "
           "this answer is the correct choice with clearer and more logical reasoning steps.")

file_to_write = open('mmlu_generate_correct_answer.jsonl', 'w')

data = read_jsonl_to_dicts('mmlu.jsonl')

i = 0
for dict_tmp in data:
    dict_to_write = {}
    dict_to_write['question_id'] = i
    dict_to_write['category'] = "reasoning"
    dict_to_write['turns'] = [prompt3 + "Question: " + dict_tmp['question'] + ", Answer: " + dict_tmp['answer']]
    final_json_str = json.dumps(dict_to_write)
    file_to_write.write(final_json_str + '\n')
    i += 1


