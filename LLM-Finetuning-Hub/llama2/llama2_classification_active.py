import argparse
import pickle
import torch
import os
import json
import copy
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from collections import defaultdict
from prompts import get_instruction_data
from llama2_classification_ppl import calculate_perplexity_teacher_cot
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from collections import Counter
from trl import SFTTrainer
import datasets
import pandas as pd
import wandb
import sys
import random

random.seed(10)

sys.path.append('../../')
from utils import read_corpus_label_rationale, training_data_read, tail_corpus, add_final_answer, \
    read_corpus_label_rationale_domain

os.environ["WANDB_PROJECT"] = "long_tail"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def find_tail_labels(args):
    original_dataset = args.dataset.split("_")[0]
    train_corpus, train_labels = training_data_read(original_dataset, args.dataset_dir)
    tail, _, _ = tail_corpus(original_dataset, train_corpus, train_labels)
    return tail


def write_stage_dataset(results_dir, train_dataset):
    if not os.path.exists(results_dir):
        # Create the directory
        os.makedirs(results_dir)

    with open(f"{results_dir}/train_data.jsonl", 'w') as outf:
        for t, l in zip(train_dataset['instructions'], train_dataset['labels']):
            results_json = {'labels': l,
                            'instructions': t}
            outf.write(f"{json.dumps(results_json)}\n")


def prepare_total_dataset(args):
    labels, rationales, corpus, predictions, domains = [], [], [], [], []
    if "r52_cot" in args.dataset:
        labels, rationales, corpus, predictions = read_corpus_label_rationale(
            f"{args.dataset_dir}/r52_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions)
        if args.method == "balanced" or args.method == "adaptive":
            labels, rationales, corpus, predictions = read_corpus_label_rationale(
                f"{args.dataset_dir}/r52_chatgpt_cot_train_composed.jsonl", labels, rationales, corpus, predictions)
        elif args.method == "random":
            pass
        else:
            raise ValueError('no such method')
    elif "reuters_cot" in args.dataset:
        labels, rationales, corpus, predictions = read_corpus_label_rationale(
            f"{args.dataset_dir}/reuters_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions)
        if args.method == "balanced" or args.method == "adaptive":
            labels, rationales, corpus, predictions = read_corpus_label_rationale(
                f"{args.dataset_dir}/reuters_chatgpt_cot_train_composed.jsonl", labels, rationales, corpus, predictions)
        elif args.method == "random":
            pass
        else:
            raise ValueError('no such method')
    elif "multichoice-qa_cot" in args.dataset:
        labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
            f"{args.dataset_dir}/multichoice-qa_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions, domains)
        if args.method == "balanced" or args.method == "adaptive":
            labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
                f"{args.dataset_dir}/multichoice-qa_chatgpt_cot_train_composed.jsonl", labels,
                rationales, corpus, predictions, domains)
        elif args.method == "random":
            pass
    elif "abstractive-qa_cot" in args.dataset:
        labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
            f"{args.dataset_dir}/abstractive-qa_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions, domains)
        if args.method == "balanced" or args.method == "adaptive":
            labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
                f"{args.dataset_dir}/abstractive-qa_chatgpt_cot_train_composed.jsonl", labels,
                rationales, corpus, predictions, domains)
        elif args.method == "random":
            pass
    elif "math_cot" in args.dataset:
        labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
            f"{args.dataset_dir}/math_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions, domains)
        if args.method == "balanced" or args.method == "adaptive":
            labels, rationales, corpus, predictions, domains = read_corpus_label_rationale_domain(
                f"{args.dataset_dir}/math_chatgpt_cot_train_composed.jsonl", labels,
                rationales, corpus, predictions, domains)
        elif args.method == "random":
            pass
    else:
        raise ValueError('no such dataset')

    print(f"Training samples:{len(rationales)}")

    # add the answer is to the end of the rationale.
    rationales = add_final_answer(predictions, rationales)

    instructions_wo_labels = get_instruction_data(args.dataset, 'inference', corpus, rationales)
    instructions_w_labels = get_instruction_data(args.dataset, 'train', corpus, rationales)

    if "r52_cot" in args.dataset:
        total_budget = args.budget * len(list(set(labels)))
    elif "reuters_cot" in args.dataset:
        total_budget = args.budget * len(list(set(labels)))
    elif "multichoice-qa_cot" in args.dataset:
        total_budget = args.budget * len(list(set(domains)))
    elif "abstractive-qa_cot" in args.dataset:
        total_budget = args.budget * len(list(set(domains)))
    elif "math_cot" in args.dataset:
        total_budget = args.budget * len(list(set(domains)))
    else:
        raise ValueError('no such dataset')

    # for dataset without domain information, we just use labels as the placeholder
    if not domains:
        domains = labels
    print(set(domains))
    total_dataset = {"instructions_wo_labels": instructions_wo_labels,
                     "instructions_w_labels": instructions_w_labels,
                     "text": corpus,
                     "rationales": rationales,
                     "labels": labels,
                     "predictions": predictions,
                     'domains': domains}

    return total_dataset, total_budget


def prepare_stage_dataset(training_number_per_stage, total_dataset, experiment_dir, current_stage_number, args,
                          method="random"):
    if method == "random":
        combined_list = list(zip(total_dataset['instructions_w_labels'], total_dataset['rationales']))
        sampled_items = random.sample(combined_list, training_number_per_stage)
        sampled_instructions, sampled_rationales = zip(*sampled_items)

        # If you need them as lists instead of tuples
        sampled_instructions = list(sampled_instructions)
        sampled_rationales = list(sampled_rationales)
    elif method == "active":
        inference_total_dataset = {"instructions": total_dataset["instructions_wo_labels"],
                                   "text": total_dataset["text"],
                                   "rationales": total_dataset["rationales"],
                                   "labels": total_dataset["labels"],
                                   "predictions": total_dataset["predictions"],
                                   "domains": total_dataset['domains']}

        output_score_list = calculate_perplexity_teacher_cot(inference_total_dataset, args.pretrained_ckpt,
                                                             experiment_dir, args.max_seq, args.batch_size)

        # Sort the list by 'ifd_ppl' in descending order and select the top N items
        top_n_results = sorted(output_score_list, key=lambda x: x['ifd_ppl'], reverse=True)[:training_number_per_stage]
        sampled_instructions = []
        for i in top_n_results:
            for i_wo_label, i_label in zip(total_dataset['instructions_wo_labels'],
                                           total_dataset['instructions_w_labels']):
                if i['instructions_wo_labels'] == i_wo_label:
                    sampled_instructions.append(i_label)
                    break

        sampled_rationales = [i["rationale"] for i in top_n_results]
        print(sampled_instructions[0])

    elif method == "downsample":
        sampled_instructions, sampled_rationales = [], []
        if "r52_cot" in args.dataset or "reuters_cot" in args.dataset:
            unique_labels = set(total_dataset['labels'])
            data_per_label = training_number_per_stage // len(unique_labels)

            # Create a dictionary to hold indices for each label
            indices_by_label = defaultdict(list)
            for index, label in enumerate(total_dataset['labels']):
                indices_by_label[label].append(index)

            # Randomly sample data_per_label data points for each label
            for label in unique_labels:
                if len(indices_by_label[label]) >= data_per_label:
                    chosen_indices = random.sample(indices_by_label[label], data_per_label)
                else:
                    # When not enough data points for a label, sample with replacement
                    chosen_indices = random.choices(indices_by_label[label], k=data_per_label)

                for index in chosen_indices:
                    sampled_instructions.append(total_dataset["instructions_w_labels"][index])
                    sampled_rationales.append(total_dataset["rationales"][index])
        elif "multichoice-qa_cot" in args.dataset or "abstractive-qa_cot" in args.dataset or "math_cot" in args.dataset:
            unique_domains = set(total_dataset['domains'])
            data_per_domain = training_number_per_stage // len(unique_domains)
            # Create a dictionary to hold indices for each domain
            indices_by_domain = defaultdict(list)
            for index, domain in enumerate(total_dataset['domains']):
                indices_by_domain[domain].append(index)

            # Randomly sample data_per_label data points for each label
            for domain in unique_domains:
                if len(indices_by_domain[domain]) >= data_per_domain:
                    chosen_indices = random.sample(indices_by_domain[domain], data_per_domain)
                else:
                    # When not enough data points for a label, sample with replacement
                    chosen_indices = random.choices(indices_by_domain[domain], k=data_per_domain)

                for index in chosen_indices:
                    sampled_instructions.append(total_dataset["instructions_w_labels"][index])
                    sampled_rationales.append(total_dataset["rationales"][index])
        else:
            raise ValueError('no such dataset')

    elif method == "downsample_active":
        inference_total_dataset = {"instructions": total_dataset["instructions_wo_labels"],
                                   "text": total_dataset["text"],
                                   "rationales": total_dataset["rationales"],
                                   "labels": total_dataset["labels"],
                                   "predictions": total_dataset["predictions"],
                                   "domains": total_dataset['domains']}
        output_score_list = calculate_perplexity_teacher_cot(inference_total_dataset, args.pretrained_ckpt,
                                                             experiment_dir, args.max_seq, args.batch_size)
        print(output_score_list[: 2])
        if "r52_cot" in args.dataset or "reuters_cot" in args.dataset:
            unique_labels = set(total_dataset['labels'])
            tail_labels = find_tail_labels(args)
            print("Tail labels: ")
            print(tail_labels)
            print(f"Number of tail labels: {len(tail_labels)}")
            data_per_label = training_number_per_stage // len(unique_labels)

            # Step 1: Group dictionaries by label
            dicts_by_label = defaultdict(list)
            for d in output_score_list:
                dicts_by_label[d['label']].append(d)

            # Step 2: Sort each group by 'ifd_ppl' in descending order
            for label in dicts_by_label:
                dicts_by_label[label].sort(key=lambda x: x['ifd_ppl'], reverse=True)

            # Step 4: Extract the top dictionaries from each group
            selected_dicts = []
            for label in dicts_by_label:

                label_dicts = dicts_by_label[label]
                num_dicts = len(label_dicts)

                # if the label is a head label, we ask LLM to annotate the question with high ifd values
                if num_dicts >= data_per_label:
                    selected_dicts.extend(label_dicts[:data_per_label])
                else:
                    repeat_factor = data_per_label // num_dicts  # Integer division to find repeat factor
                    remaining_dicts = data_per_label % num_dicts  # Find remainder for partial repeat
                    # Extend by repeated dictionaries plus any remainder by slicing
                    selected_dicts.extend(label_dicts * repeat_factor + label_dicts[:remaining_dicts])
        elif "multichoice-qa_cot" in args.dataset or "abstractive-qa_cot" in args.dataset or "math_cot" in args.dataset:
            unique_domains = set(total_dataset['domains'])
            data_per_domain = training_number_per_stage // len(unique_domains)

            # Step 1: Group dictionaries by label
            dicts_by_domain = defaultdict(list)
            for d in output_score_list:
                dicts_by_domain[d['domain']].append(d)

            # Step 2: Sort each group by 'ifd_ppl' in descending order
            for domain in dicts_by_domain:
                dicts_by_domain[domain].sort(key=lambda x: x['ifd_ppl'], reverse=True)

            # Step 4: Extract the top dictionaries from each group
            selected_dicts = []
            for domain in dicts_by_domain:

                domain_dicts = dicts_by_domain[domain]
                num_dicts = len(domain_dicts)
                print(domain)
                print(num_dicts)
                # if the domain is a head domain, we ask LLM to annotate the question with high ifd values
                if num_dicts >= data_per_domain:
                    selected_dicts.extend(domain_dicts[:data_per_domain])
                else:
                    repeat_factor = data_per_domain // num_dicts  # Integer division to find repeat factor
                    remaining_dicts = data_per_domain % num_dicts  # Find remainder for partial repeat
                    # Extend by repeated dictionaries plus any remainder by slicing
                    selected_dicts.extend(domain_dicts * repeat_factor + domain_dicts[:remaining_dicts])

        else:
            raise ValueError('no such dataset')

        sampled_instructions = []
        for i in selected_dicts:
            for i_wo_label, i_label in zip(total_dataset['instructions_wo_labels'],
                                           total_dataset['instructions_w_labels']):
                if i['instructions_wo_labels'] == i_wo_label:
                    sampled_instructions.append(i_label)
                    break
        sampled_rationales = [i["rationale"] for i in selected_dicts]
        print(sampled_instructions[0])

    elif method == "downsample_adaptive":
        inference_total_dataset = {"instructions": total_dataset["instructions_wo_labels"],
                                   "text": total_dataset["text"],
                                   "rationales": total_dataset["rationales"],
                                   "labels": total_dataset["labels"],
                                   "predictions": total_dataset["predictions"],
                                   "domains": total_dataset['domains']}

        output_score_list = calculate_perplexity_teacher_cot(inference_total_dataset, args.pretrained_ckpt,
                                                             experiment_dir, args.max_seq, args.batch_size)

        if "r52_cot" in args.dataset or "reuters_cot" in args.dataset:
            unique_labels = set(total_dataset['labels'])
            original_dataset = args.dataset.split("_")[0]
            train_corpus, train_labels = training_data_read(original_dataset, args.dataset_dir)
            # Count the occurrences of each label
            original_label_counts = Counter(random.sample(train_labels, training_number_per_stage))
            data_per_label = {}
            average_data_per_label = training_number_per_stage // len(unique_labels)
            for label in list(unique_labels):
                data_per_label[label] = int((current_stage_number / args.stage_number) * average_data_per_label + (
                        (args.stage_number - current_stage_number) / args.stage_number) * original_label_counts[label])

            # Step 1: Group dictionaries by label
            dicts_by_label = defaultdict(list)
            for d in output_score_list:
                dicts_by_label[d['label']].append(d)

            # Step 2: Sort each group by 'ifd_ppl' in descending order
            for label in dicts_by_label:
                dicts_by_label[label].sort(key=lambda x: x['ifd_ppl'], reverse=True)

            # Step 4: Extract the top dictionaries from each group
            selected_dicts = []
            for label in dicts_by_label:
                label_dicts = dicts_by_label[label]
                required_count = data_per_label[label]  # `data_per_label` is a dict specifying counts for each label
                num_dicts = len(label_dicts)

                if num_dicts >= required_count:
                    # If there are enough or more dictionaries, simply select up to the required count
                    selected_dicts.extend(label_dicts[:required_count])
                else:
                    # When there are fewer dictionaries than required, repeat them until reaching the required count
                    # Calculate how many full repeats are needed and the remainder for a partial repeat
                    full_repeats = required_count // num_dicts
                    partial_repeat_count = required_count % num_dicts
                    repeated_dicts = label_dicts * full_repeats + label_dicts[:partial_repeat_count]
                    selected_dicts.extend(repeated_dicts)

        elif "multichoice-qa_cot" in args.dataset or "abstractive-qa_cot" in args.dataset or "math_cot" in args.dataset:
            unique_domains = set(total_dataset['domains'])
            original_dataset = args.dataset.split("_")[0]
            train_question, train_labels = training_data_read(original_dataset, args.dataset_dir)
            train_labels, train_domain = train_labels

            # Count the occurrences of each domain
            original_domain_counts = Counter(random.sample(train_domain, training_number_per_stage))
            data_per_domain = {}
            average_data_per_domain = training_number_per_stage // len(unique_domains)
            for domain in list(unique_domains):
                data_per_domain[domain] = int((current_stage_number / args.stage_number) * average_data_per_domain + (
                        (args.stage_number - current_stage_number) / args.stage_number) * original_domain_counts[
                                                  domain])

            # Step 1: Group dictionaries by label
            dicts_by_domain = defaultdict(list)
            for d in output_score_list:
                dicts_by_domain[d['domain']].append(d)

            # Step 2: Sort each group by 'ifd_ppl' in descending order
            for domain in dicts_by_domain:
                dicts_by_domain[domain].sort(key=lambda x: x['ifd_ppl'], reverse=True)

            # Step 4: Extract the top dictionaries from each group
            selected_dicts = []
            for domain in dicts_by_domain:
                domain_dicts = dicts_by_domain[domain]
                required_count = data_per_domain[
                    domain]  # `data_per_domain` is a dict specifying counts for each domain
                num_dicts = len(domain_dicts)

                if num_dicts >= required_count:
                    # If there are enough or more dictionaries, simply select up to the required count
                    selected_dicts.extend(domain_dicts[:required_count])
                else:
                    # When there are fewer dictionaries than required, repeat them until reaching the required count
                    # Calculate how many full repeats are needed and the remainder for a partial repeat
                    full_repeats = required_count // num_dicts
                    partial_repeat_count = required_count % num_dicts
                    repeated_dicts = domain_dicts * full_repeats + domain_dicts[:partial_repeat_count]
                    selected_dicts.extend(repeated_dicts)

        else:
            raise ValueError('no such dataset')

        # sampled_instructions = [i["instructions_wo_labels"] + i["rationale"] for i in selected_dicts]
        sampled_instructions = []
        for i in selected_dicts:
            for i_wo_label, i_label in zip(total_dataset['instructions_wo_labels'],
                                           total_dataset['instructions_w_labels']):
                if i['instructions_wo_labels'] == i_wo_label:
                    sampled_instructions.append(i_label)
                    break
        sampled_rationales = [i["rationale"] for i in selected_dicts]
        print(sampled_instructions[0])
    else:
        raise ValueError(f'no such method: {method}')

    print(f"sampled size before filtering: {len(sampled_instructions)}")

    filtered_instructions, filtered_rationales = [], []

    # filter the instructions and rationales with wrong predictions
    for si, sr in zip(sampled_instructions, sampled_rationales):
        # Find the index of the current sampled instruction in the original dataset
        for index, (instr, pred, label) in enumerate(
                zip(total_dataset['instructions_w_labels'], total_dataset['predictions'], total_dataset['labels'])):
            if si == instr and pred == label:
                # If the instruction matches and prediction equals label, add to filtered lists
                filtered_instructions.append(si)
                filtered_rationales.append(sr)
                break  # Assuming unique instructions, break after finding the match

    added_train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={"instructions": filtered_instructions,
                  "labels": filtered_rationales}
        )
    )

    print(f"sampled size after filtering: {len(filtered_instructions)}")

    # Step 1: Convert sampled data to sets for efficient lookup
    sampled_instructions_set = set(sampled_instructions)
    sampled_rationales_set = set(sampled_rationales)

    # Step 2: Identify indexes of items to be removed
    indexes_to_remove = set()
    for index, (instruction, rationale) in enumerate(
            zip(total_dataset['instructions_w_labels'], total_dataset['rationales'])):
        if instruction in sampled_instructions_set or rationale in sampled_rationales_set:
            indexes_to_remove.add(index)

    # Create a function to remove items by index from a list
    def remove_items_by_index(lst, indexes_to_remove):
        return [item for idx, item in enumerate(lst) if idx not in indexes_to_remove]

    # Apply removal for all lists in total_dataset
    for key in total_dataset.keys():
        total_dataset[key] = remove_items_by_index(total_dataset[key], indexes_to_remove)

    print(f"Remaining total dataset number: {len(total_dataset['rationales'])}")

    return added_train_dataset, total_dataset


def get_base_model(args):
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

    # prepare base model training
    base_model = prepare_model_for_kbit_training(base_model)
    return base_model, tokenizer


def train_peft_model(args, model, tokenizer, train_dataset, peft_config, max_seq_length, results_dir, stage):
    # wandb.init(project="long_tail")

    print(f"Stage {stage} saved folder: ")
    print(results_dir)

    # wandb.init(project="long_tail")
    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=20,
        learning_rate=args.lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
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
    # wandb.finish()

    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [args.epochs, args.lora_r, args.dropout, train_loss]
        pickle.dump(run_result, handle)
    print(f"Stage {stage} Experiment over")
    return peft_model_id, results_dir


def main(args):
    stage_number = args.stage_number
    total_dataset, total_budget = prepare_total_dataset(args)
    training_number_per_stage = int(total_budget / stage_number)
    print(f"Total training number: {total_budget}")
    print(f"Stage training number: {training_number_per_stage}")
    base_model, tokenizer = get_base_model(args)
    max_seq_length = args.max_seq  # max sequence length for model and packing of the dataset
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model_id, results_dir = '', ''
    pre_train_dataset = None

    if args.method == "random":
        method_1 = "random"
        method_2 = "active"
    elif args.method == "balanced":
        method_1 = "downsample"
        method_2 = "downsample_active"
    elif args.method == "adaptive":
        method_1 = "random"
        method_2 = "downsample_adaptive"
    else:
        raise ValueError(f"no such method for active learning.")

    if "Llama-3" in args.pretrained_ckpt:
        model_name = "llama3"
    elif "llama-2" in args.pretrained_ckpt:
        model_name = "llama2"
    else:
        raise ValueError('no such model')

    for i in range(stage_number):
        if i == 0:
            added_train_dataset, total_dataset = prepare_stage_dataset(training_number_per_stage, total_dataset,
                                                                       results_dir, i, args, method=method_1)
            train_dataset = added_train_dataset
        else:
            # model = PeftModel.from_pretrained(base_model, peft_model_id)
            added_train_dataset, total_dataset = prepare_stage_dataset(training_number_per_stage, total_dataset,
                                                                       results_dir, i, args, method=method_2)
            print(f"Stage training dataset size: {len(added_train_dataset['labels'])}")
            # combine training dataset with the previous stages.
            train_instructions = added_train_dataset['instructions'] + pre_train_dataset['instructions']
            train_labels = added_train_dataset['labels'] + pre_train_dataset['labels']
            train_dataset = datasets.Dataset.from_pandas(
                pd.DataFrame(data={"instructions": train_instructions,
                                   "labels": train_labels}))
        print(f"Training data size: {len(train_dataset['labels'])}")

        # initialize a new model and train on the labeled dataset
        model = get_peft_model(base_model, peft_config)
        results_dir = f"experiments/active_learning/classification-sampleFraction-{args.train_sample_fraction}_" \
                      f"model-{model_name}_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_" \
                      f"lr-{args.lr}_dataset-{args.dataset}_method-{args.method}_stage-{i}" \
                      f"_total_{args.stage_number}_budget-{args.budget}"
        # write the prepared data to the result dir
        write_stage_dataset(results_dir, train_dataset)
        # train model from the scratch
        peft_model_id, results_dir = train_peft_model(args, model, tokenizer, train_dataset, peft_config,
                                                      max_seq_length, results_dir, stage=i)
        pre_train_dataset = copy.deepcopy(train_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="../../../llama-2-7b-hf/")
    parser.add_argument('--dataset_dir', type=str, default="/home/tonyzhou/scratch/long_tail_llm_kd/dataset/r52",
                        help='the directory contains data')
    parser.add_argument('--dataset', type=str, default="r52_cot_composed",
                        help='which dataset to fine-tune')
    parser.add_argument('--method', type=str, default="balanced",
                        help='randomly sample or balanced downsample')
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--stage_number", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--budget", required=True, type=int)

    parser.add_argument("--train_sample_fraction", default=1.0, type=float)

    args = parser.parse_args()
    main(args)
