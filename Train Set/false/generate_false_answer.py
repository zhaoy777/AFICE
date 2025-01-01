import json


def read_jsonl_to_dicts(file_path):
    """读取jsonl文件，并将每行转换为dict"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 将每行的JSON字符串转换为字典
            data.append(json.loads(line))
    return data

prompt3 = "I will give you a multiple-choice question with four options and the correct answer. Please restate the correct answer with clearer and more logical reasoning steps."

file_to_write = open('mmlu_generate_false.jsonl', 'w')

data = read_jsonl_to_dicts('../mmlu.jsonl')

i = 0
for dict_tmp in data:
    dict_to_write = {}
    dict_to_write['question_id'] = i
    dict_to_write['category'] = "reasoning"
    dict_to_map = {}
    dict_to_map['A'] = 'B'
    dict_to_map['B'] = 'C'
    dict_to_map['C'] = 'D'
    dict_to_map['D'] = 'A'
    dict_to_write['turns'] = [prompt3 + "Question: " + dict_tmp['question'] + ", Answer: " + dict_to_map[dict_tmp['answer']]]
    final_json_str = json.dumps(dict_to_write)
    file_to_write.write(final_json_str + '\n')
    i += 1

