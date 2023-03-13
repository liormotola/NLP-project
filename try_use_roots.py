from transformers import AutoTokenizer
from transformers import pipeline
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM
from project_evaluate import read_file, compute_metrics
import random
import spacy

# tokenizer = AutoTokenizer.from_pretrained("t5-base")

def create_data_with_roots(file):
    random.seed(42)
    nlp = spacy.load("en_core_web_sm")
    file_en, file_de = read_file(file)
    roots = []
    modifiers = []
    for en in file_en:
        root_str =[]
        modifiers_par = []
        parsed_en = nlp(en)
        for sen in parsed_en.sents:
            root_str.append(sen.root)
            mods = list(sen.root.children)
            if len(mods)<=2 :
                modifiers_par.append(tuple(mods))
            else:
                modifiers_par.append((mods[0],mods[1]))
        roots.append(root_str)
        modifiers.append(modifiers_par)


    data = {"German": file_de, "English": file_en,"Roots": roots,"Modifiers":modifiers}
    df = pd.DataFrame(data)
    df.to_csv("train_data_with_roots.csv",index=False)
    return df


def preprocess_function(samples):
    """
    This function performs preprocessing to data before sending it to the model. This preprocessing includes
    adding a prefix of the task
    :param samples:
    :return:
    """
    prefix = "translate German to English: "

    max_input_length = 256
    max_target_length = 256
    source_lang = "de"
    target_lang = "en"

    inputs = [prefix + sample[source_lang] for sample in samples["translation"]]
    targets = [sample[target_lang] for sample in samples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True,padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def create_translation_df_with_roots(df):
    translation = []
    for _, row in df.iterrows():

        prefix = ""
        for i, (root , mods) in enumerate(zip(row["Roots"],row["Modifiers"])):
            prefix += f"ROOT{i+1}_{root} "
            for j,mod in enumerate(mods):
                prefix += f"MOD{i+1}_{j+1}_{mod} "
        prefix += ": "
        input = prefix + row["German"]
        target = row["English"]
        raw = {'de': input, 'en': target}
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
    df= create_data_with_roots()

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
