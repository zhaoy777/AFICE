"""Generate answers with local models.

Usage:
CUDA_VISIBLE_DEVICES=0,1 python gen_model_answer_beam_sample_llama3.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import numpy as np

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

import inspect


def run_eval(
        model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        dtype,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        dtype,
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.9

        choices = []

        answer_list = []
        scores_list = [None]
        hidden_states_list = [None]
        print('1111')
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_dict = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=False,
                        temperature=0.9,
                        top_p=0.6,
                        max_new_tokens=60,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_beams=20,
                    )
                    # print('input_ids', len(input_ids[0]))
                    # print('output_dict', output_dict.keys())
                    # output_ids = output_dict["sequences"][0]
                    # print('output_ids', output_ids, len(output_ids))
                    # print(output_dict.keys())
                    scores_list[0] = output_dict["beam_scores_list"]
                    # print('scores_list', scores_list)
                    hidden_states = output_dict['hidden_states']
                    # print('hidden_states.shape', len(hidden_states), len(hidden_states[-1]), hidden_states[-1][16].shape)
                    tmp = torch.squeeze(hidden_states[-1][16])
                    hidden_states_list[0] = tmp.tolist()

                    error_msg = None
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output_ids = "Error"
                    error_msg = "Error: " + str(e) + "\n"
                    print('error_msg', error_msg)
                # turns.append(after_assistant.strip())
            # choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "model_id": model_id,
                "answer_list": answer_list,
                "scores_list": scores_list[0],
                "hidden_states_list": hidden_states_list[0],
                "choices": choices,
                "tstamp": time.time(),
            }
            # print('ans_json', ans_json)
            fout.write(json.dumps(ans_json) + "\n")
            # if error_msg is not None:
            #     fout.write(error_msg)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="confidence",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=40,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"vicuna/finetune_method/my_method/generate_train_data/{args.bench_name}/{args.file_name}.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"vicuna/finetune_method/my_method/generate_train_data/{args.bench_name}/model_answer/{args.file_name}_beam_search_20_answer.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
    )

    reorg_answer_file(answer_file)
