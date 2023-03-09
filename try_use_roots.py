from transformers import AutoTokenizer
from transformers import pipeline
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM
from project_evaluate import read_file, compute_metrics
import random

def create_data_with_roots():
    random.seed(42)
    file_en, file_de = read_file("data/train.labeled")
    final_de = []
    final_en = []
    for ger, en in zip(file_de,file_en):
        splitted_ger = ger.split("\n")[:-1]
        splitted_en = en.split("\n")[:-1]
        new_ger =""
        for ger_sen, en_sen in zip(splitted_ger,splitted_en):
            if len(en_sen.split())>4:
                sampled = random.sample(en_sen.split()[:-1],3)
            else:
                sampled = random.sample(en_sen.split(), len(en_sen.split()))
            ger_sen = " ".join(sampled) + " " + ger_sen + " "
            new_ger += ger_sen
        final_de.append(new_ger)
        final_en.append(en.replace("\n"," "))
    data = {"German": final_de, "English": final_en}
    df = pd.DataFrame(data)
    df.to_csv("train_data_with_roots.csv",index=False)
    return final_de, final_en





def read_file_unlabeled(file_path):
    file_de = []
    roots = []
    modifiers = []

    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line[:17] == "Roots in English:" :
                roots.append(line[18:])
                continue
            if line[:21] == "Modifiers in English:":
                modifiers.append(line[22:])
                continue
            if line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                cur_list = file_de
                continue
            if line:
                cur_str += line +"\n"
            else:
                cur_str+= line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_de, roots, modifiers




def create_raw_data_unlabeled(file_de,roots,modifiers):
    translation = []

    for input,root_str,modifiers_str in zip(file_de,roots,modifiers) :
        root_list = root_str.split(",")
        mods_tuples_list = modifiers_str.split(", (")
        mods_tuples_list = [" ".join(tup.strip("(").strip(")").strip().split(", ")) for tup in mods_tuples_list]
        input = input.split("\n")[:-1]
        input = [" ".join([r.strip(),m.strip(),i.strip()]) for i,r,m in zip(input,root_list,mods_tuples_list)]
        input = " ".join(input)
        raw = {'de': input}
        translation.append(raw)

    dataset = pd.DataFrame()
    dataset['translation'] = translation
    return dataset

def preprocess_function_unlabeled(examples):
    prefix = "translate German to English: "

    max_input_length = 256
    source_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)
    return model_inputs

if __name__ == '__main__':
    final_de, file_en = create_data_with_roots()

    val_file_en, val_file_de = read_file("data/val.labeled")

    lior=5

    # val_file_de,roots, modifiers = read_file_unlabeled('data/val.unlabeled')
    #
    # val_df = create_raw_data_unlabeled(val_file_de,roots, modifiers)
    #
    # vds = Dataset.from_pandas(val_df)
    #
    # val_raw_dataset = DatasetDict()
    # val_raw_dataset['validation'] = vds
    # model_checkpoint = "t5-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # val_tokenized_datasets = val_raw_dataset.map(preprocess_function_unlabeled, batched=True)
    # lior=5
