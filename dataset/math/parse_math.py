import os
import json
import csv
import numpy as np

MATH_DIR = "/nfs/turbo/coe-dkoutra/MATH/train"


writefile_path = os.path.join("/nfs/turbo/coe-dkoutra/jing/long_tail_llm_kd/dataset/math", "corpus.jsonl")
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
            try:
                label = text['solution'].split('\\boxed{')[1]
                label = label.split('$')[0]   
            except: 
                try:
                    label = text['solution'].split('\\boxed ')[1] 
                    label = label.split('$')[0]
                except:
                    breakpoint()
            results_json = {'label': label,
                            'solution': text['solution'],
                            'text': corpus,
                            'source': source,
                            'split': split}
            current_jsons.append(results_json) 

    dataset_count[cat] = len(current_jsons)
    tp.append((cat, len(current_jsons)))
    print(cat)
    print(len(current_jsons))
    jsons[cat] = current_jsons
print(dataset_count)


########Zipf distribution resampling####################
jsontoselect = {}
tout = open("json2select.json",'w')
if True:
    stdd = sorted(tp,key=lambda k:k[1],reverse=True)
    import math
    from scipy.special import zeta  
    total_number = 0
    for gp1,gp2 in stdd:
        total_number+=gp2
       
    alpha = 1.1
    s = np.random.zipf(alpha, total_number)
    counter = [0]*30
    max_sample_size = stdd[0][1]

    for item in s:
        if item<30:
            counter[item]+=1
    for item in range(len(dataset_count)):
        ratio = counter[item+1]*1.0/counter[1]
        max_sz= stdd[0][1]
        sz = stdd[item][1]
        ratio_leng = int(ratio*max_sz)
        if ratio_leng<sz:
            leng = ratio_leng
        else:
            leng = sz
        nm = stdd[item][0]
        punc_ds_idx = np.random.choice(sz,leng, replace=False)
        punc_ds_idx = sorted(punc_ds_idx)
        selected_json = []
        for j in punc_ds_idx:
            selected = jsons[nm][j]
            selected_json.append(selected)
        keyy = nm
        vall = punc_ds_idx
        jsontoselect[keyy]=vall
        print(nm,len(punc_ds_idx))
        for jj in selected_json:  
            try:
                with open(writefile_path, 'a') as outf:
                    outf.write(f"{json.dumps(jj)}\n")
            except:
                breakpoint()



