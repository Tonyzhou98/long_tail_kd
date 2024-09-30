# coding=utf-8
import sys

import datasets
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

sys.path.append('../../')
from utils import training_data_read, test_data_read

ZERO_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas: 

{newsgroup_classes}

From the above list of classes, select only one class that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the class from the given list of classes. Do not predict anything else.

### Sentence: ```{sentence}```
### Class:
"""

FEW_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 20 classes. The list of classes is provided below, where the classes are separated by commas:

{newsgroup_classes}

From the above list of classes, select only one class that the provided sentence can be classified into. Once again, only predict the class from the given list of classes. Do not predict anything else. The sentence will be delimited with triple backticks. To help you, examples are provided of sentence and the corresponding class they belong to.

{few_shot_samples}

### Sentence: ```{sentence}```
### Class:
"""

TRAINING_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

### Sentence: ```{sentence}```
### Class: {label}
"""

INFERENCE_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

### Sentence: ```{sentence}```
### Class: 
"""

TRAINING_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:{label}"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """### Sentence:{sentence} ### Class:"""

TRAINING_R52_PROMPT = """Below is a news story from the R52 dataset. Please assign a topic to this news story. \
You must select the topic from this set: copper, livestock, gold, money-fx, tea, ipi, trade, cocoa, iron-steel, \
reserves, zinc, nickel, ship, cotton, platinum, alum, strategic-metal, instal-debt, lead, housing, gnp, sugar, rubber, \
dlr, tin, interest, income, crude, coffee, jobs, meal-feed, lei, lumber, gas, nat-gas, veg-oil, orange, heat, wpi, \
cpi, earn, jet, potato, bop, money-supply, carcass, acq, pet-chem, grain, fuel, retail, cpu. 
News story:
{sentence}
Topic: 
{label}"""

INFERENCE_R52_PROMPT = """Below is a news story from the R52 dataset. Please assign a topic to this news story. \
You must select the topic from this set: copper, livestock, gold, money-fx, tea, ipi, trade, cocoa, iron-steel, \
reserves, zinc, nickel, ship, cotton, platinum, alum, strategic-metal, instal-debt, lead, housing, gnp, sugar, rubber, \
dlr, tin, interest, income, crude, coffee, jobs, meal-feed, lei, lumber, gas, nat-gas, veg-oil, orange, heat, wpi, \
cpi, earn, jet, potato, bop, money-supply, carcass, acq, pet-chem, grain, fuel, retail, cpu. 
News story:
{sentence}
Topic: 
"""

TRAINING_R52_COT_PROMPT = """Below is a news story from the R52 dataset. Please assign a topic to this news story. \
You must select the topic from this set: copper, livestock, gold, money-fx, tea, ipi, trade, cocoa, iron-steel, \
reserves, zinc, nickel, ship, cotton, platinum, alum, strategic-metal, instal-debt, lead, housing, gnp, sugar, rubber, \
dlr, tin, interest, income, crude, coffee, jobs, meal-feed, lei, lumber, gas, nat-gas, veg-oil, orange, heat, wpi, \
cpi, earn, jet, potato, bop, money-supply, carcass, acq, pet-chem, grain, fuel, retail, cpu. 
News story:
{sentence}
Let's think step by step. 
{label}"""

# INFERENCE_R52_COT_PROMPT = """Below is a news story from the R52 dataset. Please assign a topic to this news story. \
# You must select the topic from this set: copper, livestock, gold, money-fx, tea, ipi, trade, cocoa, iron-steel, \
# reserves, zinc, nickel, ship, cotton, platinum, alum, strategic-metal, instal-debt, lead, housing, gnp, sugar, rubber, \
# dlr, tin, interest, income, crude, coffee, jobs, meal-feed, lei, lumber, gas, nat-gas, veg-oil, orange, heat, wpi, \
# cpi, earn, jet, potato, bop, money-supply, carcass, acq, pet-chem, grain, fuel, retail, cpu.
# News story:
# {sentence}
# Let's think step by step.
# """

INFERENCE_R52_COT_PROMPT = """<s>human\n{sentence}<s>bot\n"""

TRAINING_REUTERS_PROMPT = """Below is a news story from the reuters dataset. Please assign a topic to this news story. \
You must select the topic from this set: acq, rubber, lead, money-supply, income, l-cattle, crude, cpu, palmkernel, \
jobs, money-fx, instal-debt, rand, castor-oil, coffee, strategic-metal, nat-gas, oat, tea, corn, yen, soy-oil, \
grain, groundnut-oil, gas, cpi, cocoa, nzdlr, soybean, rapeseed, retail, sun-meal, coconut, jet, copper, sorghum, \
carcass, heat, hog, ipi, potato, lin-oil, oilseed, alum, gnp, meal-feed, fuel, barley, ship, rape-oil, cotton-oil, \
sunseed, palm-oil, soy-meal, naphtha, nkr, trade, palladium, lei, wheat, bop, interest, earn, reserves, housing, \
veg-oil, groundnut, tin, dlr, gold, copra-cake, wpi, livestock, zinc, sugar, rye, pet-chem, dmk, dfl, orange, \
iron-steel, nickel, sun-oil, lumber, rice, propane, platinum, silver, cotton, coconut-oil.  
News story:
{sentence}
Topic: 
{label}"""

INFERENCE_REUTERS_PROMPT = """Below is a news story from the reuters dataset. Please assign a topic to this news story. \
You must select the topic from this set: acq, rubber, lead, money-supply, income, l-cattle, crude, cpu, palmkernel, \
jobs, money-fx, instal-debt, rand, castor-oil, coffee, strategic-metal, nat-gas, oat, tea, corn, yen, soy-oil, \
grain, groundnut-oil, gas, cpi, cocoa, nzdlr, soybean, rapeseed, retail, sun-meal, coconut, jet, copper, sorghum, \
carcass, heat, hog, ipi, potato, lin-oil, oilseed, alum, gnp, meal-feed, fuel, barley, ship, rape-oil, cotton-oil, \
sunseed, palm-oil, soy-meal, naphtha, nkr, trade, palladium, lei, wheat, bop, interest, earn, reserves, housing, \
veg-oil, groundnut, tin, dlr, gold, copra-cake, wpi, livestock, zinc, sugar, rye, pet-chem, dmk, dfl, orange, \
iron-steel, nickel, sun-oil, lumber, rice, propane, platinum, silver, cotton, coconut-oil. 
News story:
{sentence}
Topic: 
"""

TRAINING_REUTERS_COT_PROMPT = """Below is a news story from the reuters dataset. Please assign a topic to this news story. \
You must select the topic from this set: acq, rubber, lead, money-supply, income, l-cattle, crude, cpu, palmkernel, \
jobs, money-fx, instal-debt, rand, castor-oil, coffee, strategic-metal, nat-gas, oat, tea, corn, yen, soy-oil, \
grain, groundnut-oil, gas, cpi, cocoa, nzdlr, soybean, rapeseed, retail, sun-meal, coconut, jet, copper, sorghum, \
carcass, heat, hog, ipi, potato, lin-oil, oilseed, alum, gnp, meal-feed, fuel, barley, ship, rape-oil, cotton-oil, \
sunseed, palm-oil, soy-meal, naphtha, nkr, trade, palladium, lei, wheat, bop, interest, earn, reserves, housing, \
veg-oil, groundnut, tin, dlr, gold, copra-cake, wpi, livestock, zinc, sugar, rye, pet-chem, dmk, dfl, orange, \
iron-steel, nickel, sun-oil, lumber, rice, propane, platinum, silver, cotton, coconut-oil. 
News story:
{sentence}
Let's think step by step. 
{label}"""

# INFERENCE_REUTERS_COT_PROMPT = """Below is a news story from the reuters dataset. Please assign a topic to this news story. \
# You must select the topic from this set: acq, rubber, lead, money-supply, income, l-cattle, crude, cpu, palmkernel, \
# jobs, money-fx, instal-debt, rand, castor-oil, coffee, strategic-metal, nat-gas, oat, tea, corn, yen, soy-oil, \
# grain, groundnut-oil, gas, cpi, cocoa, nzdlr, soybean, rapeseed, retail, sun-meal, coconut, jet, copper, sorghum, \
# carcass, heat, hog, ipi, potato, lin-oil, oilseed, alum, gnp, meal-feed, fuel, barley, ship, rape-oil, cotton-oil, \
# sunseed, palm-oil, soy-meal, naphtha, nkr, trade, palladium, lei, wheat, bop, interest, earn, reserves, housing, \
# veg-oil, groundnut, tin, dlr, gold, copra-cake, wpi, livestock, zinc, sugar, rye, pet-chem, dmk, dfl, orange, \
# iron-steel, nickel, sun-oil, lumber, rice, propane, platinum, silver, cotton, coconut-oil.
# News story:
# {sentence}
# Let's think step by step.
# """


INFERENCE_REUTERS_COT_PROMPT = """<s>human\n{sentence}<s>bot\n"""

TRAINING_MULTICHOICEQA_PROMPT = """Please answer this multiple-choice question by choosing one of the given choices. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Multiple choice question:
{sentence}
Selected choice: {label}"""

INFERENCE_MULTICHOICEQA_PROMPT = """Please answer this multiple-choice question by choosing one of the given choices. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Multiple choice question:
{sentence}
Selected choice: """


TRAINING_MULTICHOICEQA_COT_PROMPT = """Please answer this multiple-choice question by choosing one of the given choices. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Multiple choice question:
{sentence}
Let's think step by step.
{label}"""

# INFERENCE_MULTICHOICEQA_COT_PROMPT = """Please answer this multiple-choice question by choosing one of the given choices. \
# If you are given a passage, please answer the question according to the passage content. \
# If the passage is not given, please answer the question directly from your knowledge.
# Multiple choice question:
# {sentence}
# Let's think step by step.
# """

INFERENCE_MULTICHOICEQA_COT_PROMPT = """<s>human\n{sentence}<s>bot\n"""


TRAINING_ABSTRACTIVEQA_PROMPT = """Please answer this question. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Question:
{sentence}
Answer: {label} """

INFERENCE_ABSTRACTIVEQA_PROMPT = """Please answer this question. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Question:
{sentence}
Answer: """


TRAINING_ABSTRACTIVEQA_COT_PROMPT = """Please answer this question. \
If you are given a passage, please answer the question according to the passage content. \
If the passage is not given, please answer the question directly from your knowledge.
Question:
{sentence}
Let's think step by step.
{label}"""

# INFERENCE_ABSTRACTIVEQA_COT_PROMPT = """Please answer this question. \
# If you are given a passage, please answer the question according to the passage content. \
# If the passage is not given, please answer the question directly from your knowledge.
# Question:
# {sentence}
# Let's think step by step.
# """

INFERENCE_ABSTRACTIVEQA_COT_PROMPT = """<s>human\n{sentence}<s>bot\n"""


TRAINING_MATH_PROMPT = """Please answer this math question by directly generating the final answer.
Question:
{sentence}
Answer: {label}"""

INFERENCE_MATH_PROMPT = """Please answer this math question by directly generating the final answer.
Question:
{sentence}
Answer: """

TRAINING_MATH_COT_PROMPT = """Please answer this math question by giving the detailed reasoning steps.
Question:
{sentence}
Let's think step by step.
{label}"""

# INFERENCE_MATH_COT_PROMPT = """Please answer this math question by giving the detailed reasoning steps.
# Question:
# {sentence}
# Let's think step by step.
# """

INFERENCE_MATH_COT_PROMPT = """<s>human\n{sentence}<s>bot\n"""

ZERO_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary:
"""

FEW_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks. To help you, examples of summarization are provided.

{few_shot_samples}

### Dialogue: ```{dialogue}```
### Summary:
"""

TRAINING_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary: {summary}
"""

TRAINING_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:{summary}"""
INFERENCE_SUMMARIZATION_PROMPT_v2 = """### Dialogue:{dialogue} ### Summary:"""

INFERENCE_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

### Dialogue: ```{dialogue}```
### Summary: 
"""


def get_instruction_data(dataset_name, mode, texts, labels):
    if "r52" in dataset_name and "r52_cot" not in dataset_name:
        if mode == "train":
            prompt = TRAINING_R52_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_R52_PROMPT
    elif dataset_name == "reuters":
        if mode == "train":
            prompt = TRAINING_REUTERS_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_REUTERS_PROMPT
    elif "r52_cot" in dataset_name:
        if mode == "train":
            prompt = TRAINING_R52_COT_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_R52_COT_PROMPT
    elif "reuters_cot" in dataset_name:
        if mode == "train":
            prompt = TRAINING_REUTERS_COT_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_REUTERS_COT_PROMPT
    elif dataset_name == "multichoice-qa":
        if mode == "train":
            prompt = TRAINING_MULTICHOICEQA_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_MULTICHOICEQA_PROMPT
    elif "multichoice-qa_cot" in dataset_name:
        if mode == "train":
            prompt = TRAINING_MULTICHOICEQA_COT_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_MULTICHOICEQA_COT_PROMPT
    elif dataset_name == "abstractive-qa":
        if mode == "train":
            prompt = TRAINING_ABSTRACTIVEQA_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_ABSTRACTIVEQA_PROMPT
    elif "abstractive-qa_cot" in dataset_name:
        if mode == "train":
            prompt = TRAINING_ABSTRACTIVEQA_COT_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_ABSTRACTIVEQA_COT_PROMPT
    elif dataset_name == "math":
        if mode == "train":
            prompt = TRAINING_MATH_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_MATH_PROMPT
    elif "math_cot" in dataset_name:
        if mode == "train":
            prompt = TRAINING_MATH_COT_PROMPT
        elif mode == "inference":
            prompt = INFERENCE_MATH_COT_PROMPT
    else:
        raise ValueError('no such dataset')

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text.replace("\n", " "),
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text.replace("\n", " "),
            )
        instructions.append(example)

    return instructions


def clean_data(texts, labels):
    label2data = {}
    clean_data, clean_labels = [], []
    for data, label in zip(texts, labels):
        if isinstance(data, str) and isinstance(label, str):
            clean_data.append(data)
            clean_labels.append(label)

            if label not in label2data:
                label2data[label] = data

    return label2data, clean_data, clean_labels


def get_classification_data_for_ft(dataset_name, dataset_dir, mode="train", train_sample_fraction=1.0, budget=-1):
    test_data, test_labels = test_data_read(dataset_name, dataset_dir)
    _, test_data, test_labels = clean_data(test_data, test_labels)
    test_instructions = get_instruction_data(dataset_name, mode, test_data, test_labels)

    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    if mode == "train":
        train_data, train_labels = training_data_read(dataset_name, dataset_dir, budget)
        label2data, train_data, train_labels = clean_data(
            train_data, train_labels
        )

        # sample n points from training data
        train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})

        if train_sample_fraction < 1:
            train_df, _ = train_test_split(
                train_df,
                train_size=train_sample_fraction,
                stratify=train_df["label"],
                random_state=42,
            )

        train_data = train_df["text"]
        train_labels = train_df["label"]

        train_instructions = get_instruction_data(dataset_name, mode, train_data, train_labels)

        train_dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(
                data={
                    "instructions": train_instructions,
                    "labels": train_labels,
                }
            )
        )

        return train_dataset, test_dataset
    else:
        return None, test_dataset


def get_samsum_data():
    samsum_dataset = load_dataset("samsum")
    train_dataset = samsum_dataset["train"]
    dialogues = train_dataset["dialogue"][:2]
    summaries = train_dataset["summary"][:2]

    few_shot_samples = ""
    for dialogue, summary in zip(dialogues, summaries):
        sample = f"Sentence: {dialogue} \n Summary: {summary} \n\n"
        few_shot_samples += sample

    return few_shot_samples
