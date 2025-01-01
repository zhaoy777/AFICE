# nohup ./script.sh 0 &
import json
import os
import sys


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


if len(sys.argv) < 2:
    sys.exit(1)  # 退出程序

turn_num = int(sys.argv[1])
directory = 'data/round' + str(turn_num)
next_directory = 'data/round' + str(turn_num + 1)
items = os.listdir(directory)
subfolders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
for subfolder in subfolders:
    print('-------' + subfolder + '-------')
    items = os.listdir(os.path.join(directory, subfolder))
    for item_name in items:
        if os.path.isdir(os.path.join(directory, subfolder, item_name)):
            continue
        print(item_name)

        name_tmp = item_name[:-6]
        answer_file_name = name_tmp + '_answer.jsonl'
        question_dicts = read_jsonl_to_dicts(directory + '/' + subfolder + '/' + item_name)
        answer_dicts = read_jsonl_to_dicts(directory + '/' + subfolder + '/' + 'llm_correct_view_5_beam_answer' + '/' + answer_file_name)

        full_directory_path = os.path.join(next_directory, subfolder)
        if not os.path.exists(full_directory_path):
            os.makedirs(full_directory_path)
        file_to_write = open(next_directory + '/' + subfolder + '/' + item_name[:-7] + str(turn_num + 1) + ".jsonl", 'w')

        for i in range(len(question_dicts)):
            question_dict = question_dicts[i]
            answer_dict = dict(answer_dicts[i])

            user_turns = question_dict['turns']
            llm_turns = question_dict['llm_turns']
            llm_turns.append(answer_dict['choices'][0]['turns'][0])

            if (len(user_turns) + len(llm_turns)) % 2 == 0:
                dict_to_write = {}
                dict_to_write['question_id'] = i
                dict_to_write['category'] = 'reasoning'
                if llm_turns[1] is None:
                    continue
                dict_to_write['turns'] = [user_turns[0], user_turns[1] + "My answer: " + llm_turns[1]]
                dict_to_write['llm_turns'] = [llm_turns[0]]
                for j in range(2, len(user_turns)):
                    dict_to_write['llm_turns'].append(user_turns[j])
                    dict_to_write['turns'].append(llm_turns[j])
                final_json_str = json.dumps(dict_to_write)
                file_to_write.write(final_json_str + '\n')
            else:
                dict_to_write = {}
                dict_to_write['question_id'] = i
                dict_to_write['category'] = 'reasoning'
                dict_to_write['turns'] = [user_turns[0]]
                dict_to_write['llm_turns'] = [llm_turns[0]]
                question_and_answer = user_turns[1]
                parts = question_and_answer.split("My answer: ")
                question_part = parts[0].strip()
                if len(parts) == 1:
                    continue
                answer_part = parts[1].strip()
                if answer_part is None or len(answer_part) < 2:
                    continue
                dict_to_write['turns'].append(question_part)
                dict_to_write['llm_turns'].append(answer_part)
                for j in range(2, len(user_turns)):
                    dict_to_write['turns'].append(llm_turns[j - 1])
                    dict_to_write['llm_turns'].append(user_turns[j])
                final_json_str = json.dumps(dict_to_write)
                file_to_write.write(final_json_str + '\n')
