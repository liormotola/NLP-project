from dependency_parsing import create_train_df, train_model, DParser
import pandas as pd
from transformers import AutoTokenizer, AdamW
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import torch.nn as nn

def tokenize_and_align_labels(examples):
    inputs = examples["sentences"]
    tokenized_inputs = tokenizer(inputs, max_length=250, truncation=True, add_special_tokens=True, padding="max_length")


    labels = []
    for i, label in enumerate(examples["heads"]):
        label = label.split()
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    true_lens = [np.count_nonzero(sen) for sen in tokenized_inputs["input_ids"]]
    tokenized_inputs["true_lens"] = true_lens

    return tokenized_inputs




def preprocess(examples ):
    inputs = examples["sentences"]
    model_inputs = tokenizer(inputs, max_length=250, truncation=True,add_special_tokens = True,padding ="max_length")

    true_lens = [np.count_nonzero(sen) for sen in model_inputs["input_ids"]]
    model_inputs["true_lens"] = true_lens
    return model_inputs

if __name__ == '__main__':

    with open("../data/HW3_train.labeled", "r") as train_file:
        train_text = train_file.read()
    with open("../data/HW3_test.labeled", "r") as test_file:
        test_text = test_file.read()

    train_df = create_train_df(train_text)
    train_df.to_csv("hw3_train_data.csv",index = False)
    test_df = create_train_df(test_text)
    test_df.to_csv("hw3_test_data.csv",index=False)

    # train_df = pd.read_csv("hw3_train_data.csv")
    # validation_df = pd.read_csv("hw3_test_data.csv")
    # df2 = pd.read_csv("test_data_new.csv")
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(train_df)
    raw_datasets = DatasetDict()
    raw_datasets['train'] = train_dataset
    raw_datasets['test'] = validation_dataset
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = raw_datasets.map(preprocess, batched=True)
    model =  DParser(model_name=model_name)
    loss_func = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    train_model(model=model,data_sets=tokenized_datasets,optimizer=optimizer,loss_func=loss_func,num_epochs=25)


