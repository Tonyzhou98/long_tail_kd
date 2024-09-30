import os
import json
import csv

TOTAL_DATASET = ['narrativeqa_dev','natural_questions_with_dpr_para', 'drop', 'qaconv', 'tweetqa']

TEST_DATASET = ['narrativeqa_dev','natural_questions_with_dpr_para', 'drop', 'qaconv', 'tweetqa']


with open("/nfs/turbo/coe-dkoutra/jing/DAMO-ConvAI/oltqa/json2select.json", 'r') as file:
    selected_dataset = json.load(file)

dataset_path = "/nfs/turbo/coe-dkoutra/jing/DAMO-ConvAI/oltqa/data_process/data"
writefile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "qa_corpus.jsonl")
writetestfile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "qa_corpus_test.jsonl")
jsons = []
dataset_count = {}
for dataset in TOTAL_DATASET:
    current_jsons = []
    current_dataset_path = os.path.join(dataset_path, dataset)
    train_selected = os.path.join(current_dataset_path, "train.tsv")
    with open(train_selected) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if len(row) > 1:
                corpus = row[0]
                label = row[1]
                source = dataset
                split = "train"
                results_json = {'label': label,
                                'text': corpus,
                                'source': source,
                                'split': split}
                current_jsons.append(results_json)  
    dataset_count[dataset] = len(current_jsons)
    print(dataset)
    print(len(current_jsons))
    for j in current_jsons:      
        with open(writefile_path, 'a') as outf:
            outf.write(f"{json.dumps(j)}\n")
print(dataset_count)
#################### test split ############################
for dataset in TEST_DATASET:
    current_jsons = []
    current_dataset_path = os.path.join(dataset_path, dataset)
    train_selected = os.path.join(current_dataset_path, "dev.tsv")
    with open(train_selected) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if len(row) > 1:
                corpus = row[0]
                label = row[1]
                source = dataset
                split = "test"
                results_json = {'label': label,
                                'text': corpus,
                                'source': source,
                                'split': split}
                current_jsons.append(results_json)  
    selected_json = current_jsons
    dataset_count[dataset] = len(selected_json)
    print(dataset)
    print(len(current_jsons))
    for j in selected_json:      
        with open(writetestfile_path, 'a') as outf:
            outf.write(f"{json.dumps(j)}\n")
print(dataset_count)