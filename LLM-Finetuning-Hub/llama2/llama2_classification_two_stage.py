import argparse
import pickle
import torch
import os
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from prompts import get_classification_data_for_ft
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import wandb

os.environ["WANDB_PROJECT"] = "long_tail"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def main(args):
    train_dataset_stage_1, _ = get_classification_data_for_ft(
        dataset_name=args.dataset_stage_1, dataset_dir=args.dataset_dir, mode="train",
        train_sample_fraction=args.train_sample_fraction, budget=args.budget
    )

    train_dataset_stage_2, _ = get_classification_data_for_ft(
        dataset_name=args.dataset_stage_2, dataset_dir=args.dataset_dir, mode="train",
        train_sample_fraction=args.train_sample_fraction, budget=args.budget
    )
    print(train_dataset_stage_1['instructions'][0])
    print(train_dataset_stage_1['labels'][0])

    print(f"Stage 1 training samples:{train_dataset_stage_1.shape}")
    print(f"Stage 2 training samples:{train_dataset_stage_2.shape}")

    # breakpoint()
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_seq_length = args.max_seq  # max sequence length for model and packing of the dataset
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for stage 1 training
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, peft_config)

    results_dir_stage_1 = f"experiments/classification-sampleFraction-{args.train_sample_fraction}_epochs-" \
                          f"{args.epoch_1}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{args.lr_1}_dataset-{args.dataset_stage_1}_" \
                          f"budget-{args.budget}"
    print("stage 1 saved folder: ")
    print(results_dir_stage_1)

    wandb.init(project="long_tail")
    training_args_stage_1 = TrainingArguments(
        output_dir=results_dir_stage_1,
        logging_dir=f"{results_dir_stage_1}/logs",
        num_train_epochs=args.epoch_1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=20,
        learning_rate=args.lr_1,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_stage_1,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args_stage_1,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    wandb.finish()

    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir_stage_1}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir_stage_1}/results.pkl", "wb") as handle:
        run_result = [
            args.epoch_1,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Stage 1 experiment over")

    # load the checkpoint in stage 1
    peft_model_id = f"{results_dir_stage_1}/assets"

    # load base LLM model and tokenizer
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(base_model, peft_model_id)

    results_dir_stage_2 = f"experiments/classification-sampleFraction-{args.train_sample_fraction}_epochs-" \
                          f"{args.epoch_2}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{args.lr_2}_dataset-{args.dataset_stage_2}_" \
                          f"budget-{args.budget}"
    print("Stage 2 saved folder: ")
    print(results_dir_stage_2)

    wandb.init(project="long_tail")
    training_args_stage_2 = TrainingArguments(
        output_dir=results_dir_stage_2,
        logging_dir=f"{results_dir_stage_2}/logs",
        num_train_epochs=args.epoch_2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=20,
        learning_rate=args.lr_2,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_stage_2,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args_stage_2,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    wandb.finish()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir_stage_2}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir_stage_2}/results.pkl", "wb") as handle:
        run_result = [
            args.epoch_2,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Stage 2 experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="../../../llama-2-7b-hf/")
    parser.add_argument('--dataset_dir', type=str, default="/home/tonyzhou/scratch/long_tail_llm_kd/dataset/r52",
                        help='the directory contains data')
    parser.add_argument('--dataset_stage_1', type=str, default="r52",
                        help='which dataset to fine-tune in the first stage')
    parser.add_argument('--dataset_stage_2', type=str, default="r52",
                        help='which dataset to fine-tune in the second stage')
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epoch_1", default=10, type=int)
    parser.add_argument("--epoch_2", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--lr_1", default=2e-4, type=float)
    parser.add_argument("--lr_2", default=1e-5, type=float)
    parser.add_argument("--budget", required=True, type=int)

    parser.add_argument("--train_sample_fraction", default=1.0, type=float)

    args = parser.parse_args()
    main(args)
