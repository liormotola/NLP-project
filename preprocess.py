import pandas as pd
import torch
from torch.utils.data import Dataset
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
        self.targets = data['English'].tolist()
        german = data["German"].tolist()
        prefix = "translate German to English: "
        self.inputs = [prefix + par for par in german]
        self.max_len_german = max([len(x.split()) for x in self.inputs])
        self.max_len_english = max([len(x.split()) for x in self.targets])
        self.model = "t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        model_inputs = self.tokenizer(self.inputs, max_length=self.max_len_german, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(self.targets, max_length= self.max_len_english, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        self.model_inputs = model_inputs


    def __getitem__(self, item):

        return {
            "input_ids": self.model_inputs["input_ids"][item],
            "attention_mask": self.model_inputs["attention_mask"][item],
            "labels": self.model_inputs["labels"][item]
        }

    def __len__(self):
        return len(self.inputs)




