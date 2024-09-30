# coding=utf-8
import argparse
import os
import pickle
import sys
import warnings
import json
import evaluate
import torch
from llama_patch import unplace_flash_attn_with_attn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from prompts import get_classification_data_for_ft
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append('../../')
from utils import verbalize_result

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def main(args):
    _, test_dataset = get_classification_data_for_ft(args.dataset, dataset_dir=args.dataset_dir, mode="inference")

    print(f"Test samples:{test_dataset.shape}")

    print(test_dataset['instructions'][: 2])
    print(test_dataset['labels'][: 2])

    experiment = args.experiment_dir
    peft_model_id = f"{experiment}/assets"

    # unpatch flash attention
    # unplace_flash_attn_with_attn()

    # load base LLM model and tokenizer
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     peft_model_id,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.bfloat16,
    #     load_in_4bit=True,
    # )
    with open(f"{experiment}/assets/adapter_config.json", 'r') as file:
        data = json.load(file)
    pretrained_ckpt = data["base_model_name_or_path"]

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # load the checkpoint for peft model
    # load base LLM model and tokenizer
    if args.base:
        # print(f"use base model: {pretrained_ckpt}")
        # model = base_model
        model = AutoModelForCausalLM.from_pretrained(
            args.mft_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"use base model: {args.mft_dir}")
    else:
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model = model.merge_and_unload()

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    if args.base:
        # result_output_file = open(f"{experiment}/test_result_base.jsonl", 'w')
        result_output_file = open(f"{args.mft_dir}/test_result.jsonl", 'w')
    else:
        result_output_file = open(f"{experiment}/test_result.jsonl", 'w')

    # for instruct, label in tqdm(zip(instructions, labels)):
    #     input_ids = tokenizer(
    #         instruct, return_tensors="pt", truncation=True
    #     ).input_ids.cuda()
    #
    #     with torch.inference_mode():
    #         try:
    #             outputs = model.generate(
    #                 input_ids=input_ids,
    #                 max_new_tokens=200,
    #                 do_sample=True,
    #                 top_p=0.95,
    #                 temperature=1e-3,
    #                 pad_token_id=tokenizer.eos_token_id,
    #             )
    #             result = tokenizer.batch_decode(
    #                 outputs.detach().cpu().numpy(), skip_special_tokens=True
    #             )[0]
    #             result = result[len(instruct):]
    #             print(result)
    #         except:
    #             print("oom")
    #             result = ""
    #             oom_examples.append(input_ids.shape[-1])
    #
    #         results_json = {'label': label,
    #                         'text': instruct,
    #                         'rationale': result}
    #         result_output_file.write(f"{json.dumps(results_json)}\n")
    #         results.append(result)
    # added_labels = labels

    # Function to process batches.

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def process_batch(batch_instructs, batch_labels):
        inputs = tokenizer(
            batch_instructs, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=400,
                do_sample=True,
                top_p=0.95,
                temperature=1e-3,
                pad_token_id=tokenizer.eos_token_id,
            )
            batch_results = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )
            # Trim the generated text to remove the input part.
            batch_results = [result[len(instruct):] for result, instruct in zip(batch_results, batch_instructs)]

        for label, instruct, result in zip(batch_labels, batch_instructs, batch_results):
            results_json = {'label': label, 'text': instruct, 'rationale': result}
            print(results_json['rationale'])
            result_output_file.write(f"{json.dumps(results_json)}\n")

        return batch_results

    # Main loop to process in batches.
    results = []
    added_labels = []

    batch_size = args.batch_size
    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructs = instructions[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        try:
            results.extend(process_batch(batch_instructs, batch_labels))
            added_labels.extend(batch_labels)
        except Exception as e:
            print(e)
            continue

    print(len(results))

    results = verbalize_result(results, args.dataset)
    result_output_file.close()

    metrics = {
        "micro_f1": f1_score(added_labels, results, average="micro"),
        "macro_f1": f1_score(added_labels, results, average="macro"),
        "precision": precision_score(added_labels, results, average="micro"),
        "recall": recall_score(added_labels, results, average="micro"),
        "accuracy": accuracy_score(added_labels, results)
    }
    print(metrics)

    save_dir = os.path.join(experiment, "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)

    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
    )
    parser.add_argument('--dataset_dir', type=str, default="/home/tonyzhou/scratch/long_tail_llm_kd/dataset/r52",
                        help='the directory contains data')
    parser.add_argument('--dataset', type=str, default="r52",
                        help='which dataset to fine-tune')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--base', action='store_true', default=False)
    parser.add_argument(
        "--mft_dir",
        default="/fs/clip-emoji/tonyzhou/long_tail_llm_kd/mft_model/math/",
    )

    args = parser.parse_args()
    main(args)
