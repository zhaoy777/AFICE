import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import gc
import os
import socket

from process_dpo_data import load_my_dataset
from datasets import Dataset


def main():
    model_name = "vicuna/vicuna-7b/vicuna-7b-v1.5-16k"
    new_model = "vicuna/vicuna-7b/finetune/prediction_entropy"

    # Load dataset
    dataset = load_my_dataset('vicuna/generate_5_response3',
                              'vicuna/train_question_answer3',
                              'vicuna/finetune_method/prediction_entropy/DPO/confidence_measure')
    # Transform list of dictionaries to dictionary of lists
    data_dict = {
        "prompt": [item['prompt'] for item in dataset],
        "chosen": [item['chosen'] for item in dataset],
        "rejected": [item['rejected'] for item in dataset],
    }

    # Create Dataset from the dictionary
    dataset = Dataset.from_dict(data_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        use_flash_attention_2=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        use_flash_attention_2=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ).eval()

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    training_args = DPOConfig(
        num_train_epochs=2,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        do_eval=True,
        per_device_eval_batch_size=1,
        adam_epsilon=1e-08,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        seed=42,
        logging_steps=100,
        save_steps=500,
        save_strategy="steps",
        output_dir="./output-dir",
        gradient_checkpointing=True,
        bf16=True,
        # fp16=True,
        remove_unused_columns=False,
    )


    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )

    torch.cuda.empty_cache()

    dpo_trainer.train()

    # Save artifacts
    dpo_trainer.model.save_pretrained(new_model)



if __name__ == "__main__":
    main()
