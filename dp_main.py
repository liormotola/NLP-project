from dependency_parsing import create_train_df, train_model, DParser
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import torch.nn as nn

def preprocess(examples ):
    inputs = examples["sentences"]
    true_lens = [len(sen.split()) for sen in examples["token_counter"]]
    targets = [[int(label) for label in cur_heads.split()] for cur_heads in examples["heads"]]
    pad_targets = np.zeros((len(inputs),250))
    for i in range(len(pad_targets)):
        pad_targets[i][:true_lens[i]] = targets[i]



    model_inputs = tokenizer(inputs, max_length=250, truncation=True,add_special_tokens = False,padding ="max_length")
    model_inputs["labels"] = pad_targets
    model_inputs["true_lens"] = true_lens
    return model_inputs

if __name__ == '__main__':

    # with open("data/HW3_train.labeled", "r") as train_file:
    #     train_text = train_file.read()
    # with open("data/HW3_test.labeled", "r") as test_file:
    #     test_text = test_file.read()
    #
    # train_df = create_train_df(train_text)
    # train_df.to_csv("hw3_train_data.csv",index = False)
    # test_df = create_train_df(test_text)
    # test_df.to_csv("hw3_test_data.csv",index=False)

    train_df = pd.read_csv("hw3_train_data.csv")
    validation_df = pd.read_csv("hw3_test_data.csv")
    # df2 = pd.read_csv("test_data_new.csv")
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(train_df)
    raw_datasets = DatasetDict()
    raw_datasets['train'] = train_dataset
    raw_datasets['test'] = validation_dataset
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = raw_datasets.map(preprocess, batched=True)
    model =  DParser(model_name=model_name)
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters())
    train_model(model=model,data_sets=tokenized_datasets,optimizer=optimizer,loss_func=loss_func,num_epochs=25)


