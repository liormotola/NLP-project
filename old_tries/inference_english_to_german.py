import torch.cuda
import pandas as pd
from old_tries.old_processing import create_translation_df
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    prefix = "translate English to German: "

    max_input_length = 180
    max_target_length = 180
    source_lang = "en"
    target_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    return model_inputs

if __name__ == '__main__':
    train_df = pd.read_csv("../train_data_new.csv")

    train_raw_df = create_translation_df(train_df)

    train_dataset = Dataset.from_pandas(train_raw_df)


    raw_datasets = DatasetDict()
    raw_datasets['train'] = train_dataset

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    # model_checkpoint = "/home/student/Final Project/Lior/t5-base-translation-from-German-to-English - checkpoints/checkpoint-22500"
    model_checkpoint = "/home/student/Final Project/Lior/t5-base-translation-from-English-to-German neww - checkpoints/checkpoint-12500"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    max_target_length = 180

    val_preds = []
    for i in tqdm(range(len(tokenized_datasets['train']["translation"][:2000])), total=2000):
        np_ins = torch.tensor(np.array([tokenized_datasets['train'][i]['input_ids']]))
        np_atten = torch.tensor(np.array([tokenized_datasets['train'][i]['attention_mask']]))

        sample_de = {'input_ids': np_ins,
                     'attention_mask': np_atten}
        translation = model.generate(**sample_de, max_length=200)
        translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
        val_preds.append(translated_text)

    with open("new_data.txt", "w", encoding="utf-8") as f:
        for pred in val_preds:
            f.write(pred+"\n")
            f.write("\n")
