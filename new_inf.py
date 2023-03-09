from transformers import AutoTokenizer
from transformers import pipeline
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
# from utils import *
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
from project_evaluate import read_file, compute_metrics


def read_file_unlabeled(file_path):
    file_de = []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line[:17] == "Roots in English:" or line[:21] == "Modifiers in English:":
                continue
            if line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                # if line == 'English:':
                #     cur_list = file_en
                # else:
                cur_list = file_de
                continue
            cur_str += line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_de




def create_raw_data_unlabeled(file_de):
    translation = []
    for input in file_de :
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
    model_checkpoint = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    val_file_de = read_file_unlabeled('data/val.unlabeled')

    val_df = create_raw_data_unlabeled(val_file_de)

    vds = Dataset.from_pandas(val_df)

    val_raw_dataset = DatasetDict()
    val_raw_dataset['validation'] = vds
    val_tokenized_datasets = val_raw_dataset.map(preprocess_function_unlabeled,batched=True)
    model_checkpoint = "/home/student/Final Project/Lior/t5-base-translation-from-German-to-English Noa/checkpoint-28000"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    translation = model.generate(input_ids = torch.tensor(val_tokenized_datasets['validation']["input_ids"]),attention_mask =torch.tensor(val_tokenized_datasets['validation']["attention_mask"]) , max_length=200)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    true_en, _ = read_file("data/val.labeled")
    tokenized_true_en = [sen.split() for sen in true_en]
    res = compute_metrics(translated_text, true_en)
    print(res)
