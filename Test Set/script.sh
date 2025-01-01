#!/bin/bash

set -e

ROUND=$1

CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name creak --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_1.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name csqa2 --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_2.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name disambiguation_qa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_3.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name gsm8k --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_4.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name logical_deduction_three_objects --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_5.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name navigate --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_6.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name penguins_in_a_table --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_7.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name prontoqa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_8.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name salient_translation_error_detection --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_9.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name sports_understanding --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_10.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name strategyqa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_11.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name temporal_sequences --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_12.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name tracking_shuffled_objects_three_objects --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_13.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_correct_view --file-name web_of_lies --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record6_14.log 2>&1

CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name creak --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_1.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name csqa2 --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_2.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name disambiguation_qa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_3.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name gsm8k --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_4.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name logical_deduction_three_objects --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_5.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name navigate --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_6.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name penguins_in_a_table --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_7.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name prontoqa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_8.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name salient_translation_error_detection --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_9.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name sports_understanding --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_10.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name strategyqa --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_11.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name temporal_sequences --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_12.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name tracking_shuffled_objects_three_objects --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_13.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python3 gen_model_answer.py --turns ${ROUND} --bench-name round${ROUND}/llm_false_view --file-name web_of_lies --model-path vicuna-7b-v1.5-16k --model-id vicuna > ./record7_14.log 2>&1