import json

import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from Data import load_data, load_confidence, CustomDataset, load_data_total_sequence, load_data_hidden_states, \
    reload_confidence, my_load_hidden_states, my_load_entropy, load_file_entropy_to_dict, load_file_hidden_states

from model import FFN, loss_function, validate, test, LSTM, Transformer, weight_validate, weighted_loss_function, \
    get_weight, Transformer2

def read_jsonl_to_dicts(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_my_file_hidden_states():
    hidden_states_dict = {'mmlu_0': {}, 'mmlu_1': {}, 'mmlu_2': {}, 'mmlu_3': {}, 'mmlu_4': {}, 'mmlu_5': {}, 'mmlu_6': {}, 'mmlu_7': {}}
    root_path = 'vicuna/finetune_method/my_method/generate_train_data/train_question/model_answer/'
    file_name_list = ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']
    # file_name_list = ['mmlu_0']

    for file_name in file_name_list:
        print('\nfile_name:', file_name, end="  ")
        datas = read_jsonl_to_dicts(root_path + file_name + '_train_question_beam_search_20_answer.jsonl')
        for content in datas:
            if content['question_id'] % 100 == 0:
                print(content['question_id'], end="  ")
            try:
                vector = []
                for j in range(20):
                    vector.append(
                        torch.tensor(content['hidden_states_list'][j]).to(dtype=torch.float32).to('cuda'))
            except Exception as e:
                # print(content['question_id'], end='@ ')
                continue
            hidden_states_dict[file_name][content['question_id']] = torch.stack(vector, dim=0)
    return hidden_states_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_states_dict = load_my_file_hidden_states()
model = Transformer(1, 4096, 64).to(device)
model.load_state_dict(torch.load('best_model.pth'))

for file_name in ['mmlu_0', 'mmlu_1', 'mmlu_2', 'mmlu_3', 'mmlu_4', 'mmlu_5', 'mmlu_6', 'mmlu_7']:

    tmp_dataset = CustomDataset(list(hidden_states_dict[file_name].values()), list(hidden_states_dict[file_name].keys()))
    train_data_loader = DataLoader(tmp_dataset, batch_size=1, shuffle=False, num_workers=0)
    file_to_write = open('vicuna/finetune_method/my_method/generate_train_data/confidence/confidence_q/' + file_name + '_entropy.jsonl', 'w')

    for batch_idx, (data, key) in enumerate(train_data_loader):
        y_pred = model(data).view(-1).to(device)
        ans_json = {
            "question_id": key.item(),
            "entropy": float(y_pred.detach().cpu().numpy()[0])
        }
        file_to_write.write(json.dumps(ans_json) + "\n")

