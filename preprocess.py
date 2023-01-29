import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score
import math
from time import time
import itertools
from transformers import AutoTokenizer

def create_train_df(text):
    """
        This method converts the text into data frame of sentences with their pos tags and their true heads (labels)

    Parameters
    ----------
    text data to convert

    Returns
    -------
    data frame of all data
    """

    couples = text.split("\n\n")[:-1]

    data = {"German":[], "English":[]}

    for couple in couples:
        german, english = couple.split("English:\n")
        german = german.split("German:\n")[-1].replace("\n", " ")
        # tokens = []
        # sentence = []
        # poss = []
        # heads = []
        # if couple == "":
        #     continue
        # for line in couple:
        #     try:
        #         splitted_line = line.split("\t")
        #         token_counter, word, pos, head = splitted_line[0],splitted_line[1], splitted_line[3],splitted_line[6]
        #         tokens.append(token_counter)
        #         sentence.append(word)
        #         poss.append(pos)
        #         heads.append(head)
        #     except:
        #         print(line)
        data["German"].append(german)
        data["English"].append(english)

    #     TODO add someting about the dependency parsing to this datatset

    df = pd.DataFrame(data)
    return df


class T5DataSet(Dataset):
    """
    add explanation
    """

    def __init__(self, data):

        data = data.dropna()
        self.english = data['English'].tolist()
        german = data["German"].tolist()
        prefix = "translate German to English: "
        self.german = [prefix + par for par in german]
        self.max_len_german = max([len(x.split()) for x in self.german])
        self.max_len_english = max([len(x.split()) for x in self.english])
        self.model = "t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        # for german, english in zip(self.german, self.english):
        #
        #     cur_rep = []
        #     cur_rep_pos = []
        #     cur_heads = [int(label) for label in cur_heads.split()]
        #     cur_pos = cur_pos.split()
        #     for i, word in enumerate(sen.split()):
        #         if word in embedding_model.key_to_index:
        #             vec = embedding_model[word]
        #         elif word.lower() in embedding_model.key_to_index:
        #             vec = embedding_model[word.lower()]
        #         else:
        #             vec = embedding_model["UNK"]
        #
        #         pos = cur_pos[i]
        #         if pos in pos2idx:
        #             cur_rep_pos.append(pos2idx[pos])
        #
        #         else:
        #             cur_rep_pos.append(pos2idx["UNK"])
        #
        #         cur_rep.append(vec)
        #
        #     pad = np.zeros((max_sen_len, 300))
        #
        #     self.true_lens.append(len(cur_rep))
        #     if len(cur_rep) < max_sen_len:
        #         pad[:len(cur_rep)] = cur_rep
        #         cur_rep = pad
        #         pad_pos = np.zeros(max_sen_len)
        #         pad_pos[:len(cur_rep_pos)] = cur_rep_pos
        #         cur_rep_pos = pad_pos
        #         pad_head = np.zeros(max_sen_len)
        #         pad_head[:len(cur_heads)] = cur_heads
        #         cur_heads = pad_head
        #
        #     heads.append(cur_heads)
        #     representation.append(cur_rep)
        #     poss.append(cur_rep_pos)
        #
        # heads = np.stack(heads)
        # self.heads = heads
        # poss = np.stack(poss)
        # self.poss = poss
        # representation = np.stack(representation)
        # self.embedded_sens = representation
        # self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):

        source_text = self.german[item]
        target_text = self.english[item]
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_len_german,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.max_len_english,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "labels": target_ids.to(dtype=torch.long)
        }

    def __len__(self):
        return len(self.german)




