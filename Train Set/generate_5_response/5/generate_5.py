import json


def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


prompt1 = ("Below I will give you a question, along with two viewpoints on that question, "
           "presented in the following format: \nQuestion: {...}\nViewpoint 1: {...}\nViewpoint 2: {...}\n")
prompt2 = ("Assuming you originally held Viewpoint 1, now please generate the following response: Admit your mistake and agree with Viewpoint 2.")

category_name = ['mmlu']
category_id = 0
file_to_write = open('generate_5_5_' + category_name[category_id] + '.jsonl', 'w', encoding='utf-8')
origin_questions = read_jsonl_to_dicts('../../' + category_name[category_id] + '.jsonl')
true_answers = read_jsonl_to_dicts(
    '../../ture_answer/' + category_name[category_id] + '_beam_sample_question_answer.jsonl')
correct_answers = read_jsonl_to_dicts(
    '../../correct_answer/' + category_name[category_id] + '_generate_correct_answer.jsonl')
false_answers = read_jsonl_to_dicts('../../false_answer/' + category_name[category_id] + '_generate_false_answer.jsonl')
judge_results = read_jsonl_to_dicts(
    '../../generate_5_response_judge/' + category_name[category_id] + '_beam_sample_question_answer_judge.jsonl')


correct_answers_dict = {}
for item in correct_answers:
    correct_answers_dict[item['question_id']] = item
keys_list = correct_answers_dict.keys()
for judge_result in judge_results:
    id = int(judge_result['question_id'])
    tag = judge_result['tag']
    question = origin_questions[id]['question']
    model_answer = true_answers[id]['choices'][0]['turns'][0]
    false_answer = false_answers[id]['choices'][0]['turns'][0]
    if id in keys_list:
        correct_answer = correct_answers_dict[id]['choices'][0]['turns'][0]
    else:
        continue
    if tag == 'correct':
        dict_to_write = {}
        dict_to_write['question_id'] = id
        dict_to_write['tag'] = 'correct'
        dict_to_write['category'] = "reasoning"
        dict_to_write['turns'] = [
            prompt1 + "\nQuestion: " + question + "\nViewpoint 1: " + model_answer + "\nViewpoint 2: " + false_answer + '\n' + prompt2]
        final_json_str = json.dumps(dict_to_write)
        file_to_write.write(final_json_str + '\n')
    else:
        dict_to_write = {}
        dict_to_write['question_id'] = id
        dict_to_write['tag'] = 'wrong'
        dict_to_write['category'] = "reasoning"
        dict_to_write['turns'] = [
            prompt1 + "\nQuestion: " + question + "\nViewpoint 1: " + model_answer + "\nViewpoint 2: " + correct_answer + '\n' + prompt2]
        final_json_str = json.dumps(dict_to_write)
        file_to_write.write(final_json_str + '\n')
