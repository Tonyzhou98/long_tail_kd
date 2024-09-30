import os
import json
import csv
import numpy as np

MATH_DIR = "/nfs/turbo/coe-dkoutra/MATH/test"


writefile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/math", "corpus_test.jsonl")
jsons = {}
dataset_count = {}
tp = []
### load all math files
categories = os.listdir(MATH_DIR)
category_count = {}
for cat in categories:
    cat_path = os.path.join(MATH_DIR, cat)
    files = os.listdir(cat_path)
    current_jsons = []
    for f in files: 
        with open(os.path.join(cat_path, f), 'r') as fle:
            text = json.load(fle)
            corpus = text['problem']
            source = text['type']
            split = "train"
            if 'boxed' in text['solution']:
                ans = text['solution'].split('boxed')[-1]
                if (ans[0] == '{'):
                    stack = 1
                    a = ''
                    for c in ans[1:]:
                        if (c == '{'):
                            stack += 1
                            a += c
                        elif (c == '}'):
                            stack -= 1
                            if (stack == 0): break
                            a += c
                        else:
                            a += c
            else:
                breakpoint
            label = a
            if len(label) > 0:
                results_json = {'label': label,
                                'org': text['solution'],
                                'text': corpus,
                                'source': source,
                                'split': split}
                current_jsons.append(results_json) 

    dataset_count[cat] = len(current_jsons)
    tp.append((cat, len(current_jsons)))
    print(cat)
    print(len(current_jsons))
    jsons[cat] = current_jsons
    for j in current_jsons:      
        with open(writefile_path, 'a') as outf:
            outf.write(f"{json.dumps(j)}\n")
print(dataset_count)

