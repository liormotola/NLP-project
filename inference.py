from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM
from try_use_roots import read_file_unlabeled_with_roots

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


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
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                cur_list = file_de
                continue
            cur_str += line + ' '
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
    prefix = "translate German to English "

    max_input_length = 256
    source_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)
    return model_inputs


def create_translation_df_unlabeled_with_roots(file_de,roots,modifiers):
    translation = []

    for input,root_str,modifiers_str in zip(file_de,roots,modifiers) :
        root_list = root_str.split(",")
        mods_tuples_list = modifiers_str.split(", (")
        mods_tuples_list = [(tup.strip("(").strip(")").strip().split(", ")) for tup in mods_tuples_list]
        prefix = ""
        for i, (root, mods) in enumerate(zip(root_list, mods_tuples_list)):
            prefix += f"ROOT{i + 1}_{root.strip()} "
            for j, mod in enumerate(mods):
                prefix += f"MOD{i + 1}_{j + 1}_{mod.strip()} "
        prefix += ": "
        input = prefix + input
        raw = {'de': input}
        translation.append(raw)

    dataset = pd.DataFrame()
    dataset['translation'] = translation
    return dataset




def perform_inference(de_input_list,model_checkpoint):

    val_df = create_raw_data_unlabeled(de_input_list)
    val_dataset = Dataset.from_pandas(val_df)
    val_raw_dataset = DatasetDict()
    val_raw_dataset['validation'] = val_dataset
    val_tokenized_datasets = val_raw_dataset.map(preprocess_function_unlabeled, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    translation = model.generate(input_ids=torch.tensor(val_tokenized_datasets['validation']["input_ids"]),
                                 attention_mask=torch.tensor(val_tokenized_datasets['validation']["attention_mask"]),
                                 max_length=200)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return translated_text

def perform_inference_with_roots(file_path,model_checkpoint):
    file_de, roots, modifiers= read_file_unlabeled_with_roots(file_path)
    val_df = create_translation_df_unlabeled_with_roots(file_de=file_de,roots=roots,modifiers=modifiers)
    val_dataset = Dataset.from_pandas(val_df)
    val_raw_dataset = DatasetDict()
    val_raw_dataset['validation'] = val_dataset
    val_tokenized_datasets = val_raw_dataset.map(preprocess_function_unlabeled, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    translation = model.generate(input_ids=torch.tensor(val_tokenized_datasets['validation']["input_ids"]),
                                 attention_mask=torch.tensor(val_tokenized_datasets['validation']["attention_mask"]),
                                 max_length=200)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return file_de, translated_text


if __name__ == '__main__':
    perform_inference_with_roots("data/val.unlabeled",model_checkpoint="/home/student/Final Project/Lior/t5-base-translation-from-German-to-English-with_15e5_and_roots/checkpoint-14000")