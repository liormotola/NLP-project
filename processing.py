import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def create_train_df(text):
    """
    This method converts the text into data frame of sentences - 2 columns - german and english

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
        english = english.replace("\n", " ")
        data["German"].append(german)
        data["English"].append(english)

    df = pd.DataFrame(data)
    return df



def create_translation_df(df):
    translation = []
    for input, target in zip(df.German.values, df.English.values):
        raw = {'de': input, 'en': target}
        translation.append(raw)

    dataset = pd.DataFrame()
    dataset['translation'] = translation
    return dataset

def create_translation_df_from_list(german,english):
    translation = []
    for input, target in zip(german, english):
        raw = {'de': input, 'en': target}
        translation.append(raw)

    dataset = pd.DataFrame()
    dataset['translation'] = translation
    return dataset

def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]

    return preds, labels



