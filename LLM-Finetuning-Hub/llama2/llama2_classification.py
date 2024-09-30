import argparse
import pickle
import torch
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from prompts import get_classification_data_for_ft
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def main(args):
    train_dataset, test_dataset = get_classification_data_for_ft(
        dataset_name=args.dataset, dataset_dir=args.dataset_dir, mode="train",
        train_sample_fraction=args.train_sample_fraction, budget=args.budget
    )
    print(train_dataset['instructions'][0])
    print(train_dataset['labels'][0])

    print(f"Sample fraction:{args.train_sample_fraction}")
    print(f"Training samples:{train_dataset.shape}")

    print(f"Training samples:{train_dataset.shape}")
    print(f"Test samples:{test_dataset.shape}")

    # breakpoint()
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    if "Llama-3" in args.pretrained_ckpt:
        model_name = "llama3"
    elif "llama-2" in args.pretrained_ckpt:
        model_name = "llama2"
    else:
        raise ValueError('no such model')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    results_dir = f"experiments/classification-sampleFraction-{args.train_sample_fraction}_model-{model_name}_" \
                  f"epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_" \
                  f"dataset-{args.dataset}_budget-{args.budget}"
    max_seq_length = args.max_seq  # max sequence length for model and packing of the dataset

    print("saved folder: ")
    print(results_dir)

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="../../../Meta-Llama-3-8B/")
    parser.add_argument('--dataset_dir', type=str, default="/home/tonyzhou/scratch/long_tail_llm_kd/dataset/r52",
                        help='the directory contains data')
    parser.add_argument('--dataset', type=str, default="r52",
                        help='which dataset to fine-tune')
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=1.0, type=float)
    parser.add_argument("--budget", required=True, type=int)

    args = parser.parse_args()
    main(args)
