import os
import json
import csv
import random

corpus_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "qa_corpus.jsonl")
test_corpus_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "qa_corpus_test.jsonl")

writefile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "corpus.jsonl")
writetestfile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/abstractive-qa", "test_corpus.jsonl")
jsons = []
test_jsons = []
import re

def remove_special_characters(input_string):
    # This will replace anything that is not a letter or number with an empty string
    result = re.sub(r'[^A-Za-z0-9]', '', input_string)
    return result

### reformat ####
with open(corpus_path, 'r') as inf:
    for line in inf:        
        data = json.loads(line.strip())
        split_text = data['text'].split('\\n')
        question = split_text[0]
        if len(split_text) > 1:
            passage = split_text[1]
        else:
            passage = None
            breakpoint()

        if passage is None: 
            breakpoint()
            reformat_text = "Question:" + question + '\n'
        else:
            reformat_text = "Passage:" + passage + '\n' + "Question:" + question + '\n'
        results_json = {'label': data['label'],
                        'text': reformat_text,
                        'source': data['source'],
                        'split':'train'}
        jsons.append(results_json)  

    total_count = 10000
    # Sample n items
    sampled_items = random.sample(jsons, total_count)
    # count stats
    dataset_count = {}
    for item in sampled_items:
        if item['source'] in dataset_count:
            dataset_count[item['source']] += 1
        else:
            dataset_count[item['source']] = 1
    print(dataset_count)

for j in sampled_items:      
    with open(writefile_path, 'a') as outf:
        outf.write(f"{json.dumps(j)}\n")

print(len(jsons))

#### reformat ####
with open(test_corpus_path, 'r') as inf:
    for line in inf:
        data = json.loads(line.strip())
        split_text = data['text'].split('\\n')
        question = split_text[0]
        if len(split_text) > 1:
            passage = split_text[1]
        else:
            passage = None
            breakpoint()

        if passage is None: 
            breakpoint()
            reformat_text = "Question:" + question + '\n'
        else:
            reformat_text = "Passage:" + passage + '\n' + "Question:" + question + '\n'
        results_json = {'label': data['label'],
                        'text': reformat_text,
                        'source': data['source'],
                        'split':'test'}
        test_jsons.append(results_json)  

    total_count = 10000
    # Sample n items
    sampled_items = random.sample(test_jsons, total_count)
    # count stats
    dataset_count = {}
    for item in sampled_items:
        if item['source'] in dataset_count:
            dataset_count[item['source']] += 1
        else:
            dataset_count[item['source']] = 1
    print(dataset_count)


for j in sampled_items:      
    with open(writetestfile_path, 'a') as outf:
        outf.write(f"{json.dumps(j)}\n")

print(len(test_jsons))