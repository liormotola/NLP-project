import torch.cuda
import pandas as pd
from old_tries.old_processing import create_translation_df
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

model =AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def german_to_spanish(examples):
    prefix = "translate German to English: "

    max_input_length = 180
    source_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    return model_inputs

train_df = pd.read_csv("../train_data_new.csv")
train_raw_df = create_translation_df(train_df)
train_dataset = Dataset.from_pandas(train_raw_df)

raw_datasets = DatasetDict()
raw_datasets['train'] = train_dataset
tokenized_datasets = raw_datasets.map(german_to_spanish, batched=True)


new_german = []
for i in tqdm(range(len(tokenized_datasets['train']["translation"][:1000])), total=1000):
    np_ins = torch.tensor(np.array([tokenized_datasets['train'][i]['input_ids']]))
    np_atten = torch.tensor(np.array([tokenized_datasets['train'][i]['attention_mask']]))

    sample_de = {'input_ids': np_ins,
                 'attention_mask': np_atten}
    spanish = model.generate(**sample_de, max_length=250)
    spanish_from_german = tokenizer.batch_decode(spanish, skip_special_tokens=True)[0]

    spanish_from_german = "translate English to German: " + spanish_from_german

    tokenized_spanish = tokenizer(spanish_from_german, return_tensors="pt", max_length=250, truncation=True)
    genrated_german = model.generate(**tokenized_spanish, max_length =250)
    german = tokenizer.batch_decode(genrated_german,skip_special_tokens=True)[0]

    new_german.append(german)
print(new_german)
with open("flant5_new_german.txt", "w") as f:
  for line in new_german:
    f.write(line+'\n')
    f.write('\n')