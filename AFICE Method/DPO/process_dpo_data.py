import json
import re
from transformers import AutoTokenizer




def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_id2index_dict(list_tmp):
    id2index_dict = {}
    for i in range(len(list_tmp)):
        id2index_dict[list_tmp[i]['question_id']] = i
    return id2index_dict


def process_answer(answer):
    modified_text = answer.replace("Viewpoint 1", "The viewpoint I just provided").replace("Viewpoint 2",
                                                                                           "Your viewpoint")
    return modified_text


def chatml_format(question, model_answer, user_opinion, chosen, rejected):
    system = ""
    prompt = ("<|im_start|>User:\n" + question + "<|im_end|>\n<|im_start|>Assistant:\n" + model_answer + ("<|im_end|>\n"
                                                                                                        "<|im_start"
                                                                                                        "|>User:\n") +
              user_opinion + "<|im_end|>\n<|im_start|>Assistant:\n")

    chosen = chosen + "<|im_end|>\n"

    rejected = rejected + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def examine_answer_list_length(answer_list):
    return_list = []
    for answer in answer_list:
        if len(answer) < 8:
            continue
        else:
            return_list.append(answer)
    return return_list


def pairwise_pairing(question, model_origin_answer, user_opinion, chosen_list, rejected_list, tokenizer, list_for_calculate):
    formal_data_list = []
    for chosen in chosen_list:
        for rejected in rejected_list:
            return_dict = chatml_format(question, model_origin_answer, user_opinion, chosen, rejected)

            prompt_tokens = tokenizer.tokenize(return_dict['prompt'])
            chosen_tokens = tokenizer.tokenize(return_dict['chosen'])
            reject_tokens = tokenizer.tokenize(return_dict['rejected'])

            list_for_calculate[0] += len(prompt_tokens)
            list_for_calculate[1] += len(chosen_tokens)
            list_for_calculate[2] += len(reject_tokens)

            if len(prompt_tokens) < 512 and len(chosen_tokens) < 512 and len(reject_tokens) < 512:
                formal_data_list.append(return_dict)
                list_for_calculate[3] += 1
    return formal_data_list


def load_my_dataset(response_5_path, model_answer_path, confidence_path):
    '''
    answer和category都是残缺的
    1.先求出两个集合的id的交集
    2.然后从这个交集里面顺序遍历
    3.如果category=0，则从1、2、3中选择正例，从4、5中选择负例；
      如果category=1，则从2、3、4中选择正例，从1、5中选择负例；
      如果category=2，则从3、4、5中选择正例，从1、2中选择负例；
    4.问题怎么构建呢？
      用户提问 + 模型回答 + 用户观点 + 模型观点
      用户提问 从response_5_path的问题中提取
      模型回答 从model_answer_path中提取
      用户观点 从response_5_path的问题中提取
      模型观点 从response_5_path的回答中提取

    :param response_5_path: 5种回答类型的存放目录
    :param confidence_path: confidence数据的存放目录
    :return: 返回整个dataset
    '''

    total_data_list = []
    dataset_name_list = ['mmlu']
    # dataset_name_list = ['gsm8k']
    for dataset_name in dataset_name_list:
        answers1 = read_jsonl_to_dicts(
            response_5_path + '/1/answer/' + 'generate_5_1_' + dataset_name + '_answer.jsonl')
        answers2 = read_jsonl_to_dicts(
            response_5_path + '/2/answer/' + 'generate_5_2_' + dataset_name + '_answer.jsonl')
        answers3 = read_jsonl_to_dicts(
            response_5_path + '/3/answer/' + 'generate_5_3_' + dataset_name + '_answer.jsonl')
        answers4 = read_jsonl_to_dicts(
            response_5_path + '/4/answer/' + 'generate_5_4_' + dataset_name + '_answer.jsonl')
        answers5 = read_jsonl_to_dicts(
            response_5_path + '/5/answer/' + 'generate_5_5_' + dataset_name + '_answer.jsonl')

        categories = read_jsonl_to_dicts(confidence_path + '/' + dataset_name + '_prediction_category.jsonl')
        model_answers = read_jsonl_to_dicts(model_answer_path + '/' + dataset_name + '_train_question_answer.jsonl')

        questions = read_jsonl_to_dicts(response_5_path + '/1/' + 'generate_5_1_' + dataset_name + '.jsonl')

        answers1_id2index_dict = get_id2index_dict(answers1)
        answers2_id2index_dict = get_id2index_dict(answers2)
        answers3_id2index_dict = get_id2index_dict(answers3)
        answers4_id2index_dict = get_id2index_dict(answers4)
        answers5_id2index_dict = get_id2index_dict(answers5)
        categories_id2index_dict = get_id2index_dict(categories)
        questions_id2index_dict = get_id2index_dict(questions)
        model_answers_id2index_dict = get_id2index_dict(model_answers)

        answer_id_list = [data['question_id'] for data in answers1]
        category_id_list = [data['question_id'] for data in categories]
        model_answer_id_list = [data['question_id'] for data in model_answers]
        questions_id_list = [data['question_id'] for data in questions]

        intersection = set(answer_id_list) & set(category_id_list) & set(model_answer_id_list) & set(questions_id_list)

        intersection_list = sorted(list(intersection))

        tokenizer = AutoTokenizer.from_pretrained('vicuna/vicuna-7b/vicuna-7b-v1.5-16k')

        list_for_calculate = [0, 0, 0, 0]

        for true_id in intersection_list:
            question_and_user_opinion = questions[questions_id2index_dict[true_id]]['turns'][0]
            true_questions = re.findall(r'Question: (.*?)(?=\nViewpoint 1:)', question_and_user_opinion, re.S)
            if len(true_questions) >= 2:
                true_question = true_questions[1]
            else:
                print("Less than two questions were found.")
                continue
            true_question = true_question.strip()
            viewpoints2 = re.findall(r'Viewpoint 2: (.*?)(?=Viewpoint 2:|Assuming)', question_and_user_opinion, re.S)
            user_opinion = viewpoints2[1]
            model_origin_answer = model_answers[model_answers_id2index_dict[true_id]]['choices'][0]['turns'][0]
            category = int(categories[categories_id2index_dict[true_id]]['category'])

            final_answer1 = process_answer(answers1[answers1_id2index_dict[true_id]]['choices'][0]['turns'][0])
            final_answer2 = process_answer(answers2[answers2_id2index_dict[true_id]]['choices'][0]['turns'][0])
            final_answer3 = process_answer(answers3[answers3_id2index_dict[true_id]]['choices'][0]['turns'][0])
            final_answer4 = process_answer(answers4[answers4_id2index_dict[true_id]]['choices'][0]['turns'][0])
            final_answer5 = process_answer(answers5[answers5_id2index_dict[true_id]]['choices'][0]['turns'][0])

            if category == 0:
                chosen_list = examine_answer_list_length([final_answer1, final_answer2, final_answer3])
                rejected_list = examine_answer_list_length([final_answer4, final_answer5])
            elif category == 1:
                chosen_list = examine_answer_list_length([final_answer2, final_answer3, final_answer4])
                rejected_list = examine_answer_list_length([final_answer1, final_answer5])
            else:
                chosen_list = examine_answer_list_length([final_answer3, final_answer4, final_answer5])
                rejected_list = examine_answer_list_length([final_answer1, final_answer2])

            if len(chosen_list) >= 1 and len(rejected_list) >= 1:
                total_data_list += pairwise_pairing(true_question, model_origin_answer, user_opinion, chosen_list,
                                                    rejected_list, tokenizer, list_for_calculate)

    # print('length', len(total_data_list))
    # print('avg', list_for_calculate[0] / len(total_data_list), list_for_calculate[1] / len(total_data_list), list_for_calculate[2] / len(total_data_list), 'correct', list_for_calculate[3])

    return total_data_list