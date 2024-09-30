import os
import sys
import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from tqdm import tqdm
from prompts import get_instruction_data
import torch
from torch.multiprocessing import Process, Queue, set_start_method
from typing import List

sys.path.append('../../')
from utils import read_corpus_label_rationale


# Used to get the ppl and emb for the whole output
def get_perplexity_whole_text(tokenizer, model, text, max_length):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    except Exception as e:
        print(e)
        return 0, 0


# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_part_text(tokenizer, model, whole_text, output, max_length):
    try:
        input_ids = tokenizer.encode(whole_text, return_tensors="pt", truncation=True, max_length=max_length).cuda()
        start_index = whole_text.rfind(output)
        start_token = len(tokenizer.encode(whole_text[:start_index]))

        labels = input_ids.clone()
        # -100 is the default ignore_index in PyTorch’s CrossEntropyLoss.
        # Any token with a label of -100 will be ignored in loss computation.

        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    except Exception as e:
        print(e)
        return 0, 0


def calculate_perplexity_teacher_cot(train_dataset, pretrained_ckpt, experiment_dir, max_seq, batch_size=16):
    """
    Perform text generation on a specific device (GPU) and write outputs to a queue.
    """

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # load the checkpoint for peft model
    peft_model_id = f"{experiment_dir}/assets"
    # load base LLM model and tokenizer
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length = max_seq  # max sequence length for model and packing of the dataset
    model.eval()

    output_list = []
    inference_whole_text, inference_instructions, inference_labels, \
    inference_predictions, inference_rationale, inference_text, inference_domains = [], [], [], [], [], [], []

    print(f"Finish loading the model on gpu")

    with torch.inference_mode():
        instructions = train_dataset['instructions']
        for i in tqdm(range(0, len(instructions), batch_size)):
            batch_instructs = instructions[i:i + batch_size]
            inputs = tokenizer(
                batch_instructs, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    top_p=0.5,
                    temperature=1e-3,
                    pad_token_id=tokenizer.eos_token_id,
                )
                batch_whole_text = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )
                # print(batch_whole_text)
                inference_whole_text.extend(batch_whole_text)
                inference_labels.extend(train_dataset['labels'][i:i + batch_size])
                inference_text.extend(train_dataset['text'][i:i + batch_size])
                inference_predictions.extend(train_dataset['predictions'][i:i + batch_size])
                inference_rationale.extend(train_dataset['rationales'][i:i + batch_size])
                inference_instructions.extend(train_dataset['instructions'][i:i + batch_size])
                inference_domains.extend(train_dataset['domains'][i:i + batch_size])
            except Exception as e:
                print(e)
                continue

    print(inference_whole_text[: 1])

    with torch.inference_mode():
        for instruct_i, rationale_i, text_i, label_i, prediction_i, whole_text_i, domain_i in tqdm(
                zip(inference_instructions, inference_rationale, inference_text, inference_labels,
                    inference_predictions, inference_whole_text, inference_domains)):
            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True,
                                                    max_length=max_length).cuda()

            output_i = whole_text_i[len(instruct_i):]
            # print(output_i)

            instruct_i_len = instruct_i_input_ids.shape[1]
            ppl_out_condition, loss_out_condition = get_perplexity_part_text(tokenizer, model, whole_text_i,
                                                                             output_i, max_length)
            ppl_out_alone, loss_out_alone = get_perplexity_whole_text(tokenizer, model, output_i,
                                                                      max_length - instruct_i_len + 1)
            results_json = {
                'text': text_i,
                'rationale': rationale_i,
                'student_output': output_i,
                'instructions_wo_labels': instruct_i,
                'label': label_i,
                'prediction': prediction_i,
                'ppl_condition': ppl_out_condition,
                'ppl_A_direct': ppl_out_alone,
                'ifd_ppl': ppl_out_condition / ppl_out_alone,
                'domain': domain_i
            }

            output_list.append(results_json)
    return output_list


def ppl_generation_and_write_to_file(train_dataset, output_file_path, args):
    output_list = calculate_perplexity_teacher_cot(train_dataset, args.pretrained_ckpt, args.experiment_dir,
                                                   args.max_seq, args.batch_size)

    # Collect all generated texts from the queue and write them to a file
    with open(output_file_path, 'w') as f:
        for generated_text in output_list:
            f.write(f'{generated_text}\n')


def main(args):
    dataset_name = args.dataset
    dataset_dir = args.dataset_dir

    if dataset_name == "r52_cot_composed":
        labels, rationales, corpus, predictions = [], [], [], []
        labels, rationales, corpus, predictions = read_corpus_label_rationale(
            f"{args.dataset_dir}/r52_chatgpt_cot_train.jsonl", labels,
            rationales, corpus, predictions)
        labels, rationales, corpus, predictions = read_corpus_label_rationale(
            f"{args.dataset_dir}/r52_chatgpt_cot_train_composed.jsonl", labels, rationales, corpus, predictions)
        print(f"Training samples:{len(rationales)}")
        ppl_path = f"{dataset_dir}/r52_chatgpt_cot_train_composed_ppl.jsonl"
    else:
        raise ValueError('no such dataset')

    if os.path.exists(ppl_path):
        annotated_texts = []
        with open(ppl_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                annotated_texts.append(data['text'])

        zipped_data = zip(corpus, rationales, predictions, labels)
        filtered_data = [(t, r, p, l) for t, r, p, l in zipped_data if t not in annotated_texts]
        corpus, rationales, predictions, labels = zip(*filtered_data)
        corpus = list(corpus)
        rationales = list(rationales)
        predictions = list(predictions)
        labels = list(labels)

    instructions = get_instruction_data(dataset_name, 'inference', corpus, rationales)

    train_dataset = {"instructions": instructions,
                     "text": corpus,
                     "rationales": rationales,
                     "labels": labels,
                     "predictions": predictions,
                     'domains': labels}

    print(f"dataset to calculate ppl: {len(train_dataset['text'])}")

    ppl_generation_and_write_to_file(train_dataset, ppl_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="../../../llama-2-7b-hf/")
    parser.add_argument(
        "--experiment_dir",
        default="experiments/classification-sampleFraction-0.1_epochs-5_rank-8_dropout-0.1",
    )
    parser.add_argument('--dataset', type=str, default="r52_cot_diverse_explain",
                        help='which dataset to fine-tune')
    parser.add_argument('--dataset_dir', type=str, default="/home/tonyzhou/scratch/long_tail_llm_kd/dataset/r52",
                        help='the directory contains data')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_seq", default=1024, type=int)

    args = parser.parse_args()
    main(args)
