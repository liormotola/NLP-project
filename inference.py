from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM
from processing import read_file_unlabeled_with_roots

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def read_file_unlabeled(file_path):
    """
    Given a file in unlabeled format, parsing the file and returns only the german sentences
    :param file_path: path of file to parse
    :return: list of sentences in german
    """
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
    """
    preparing data of unlabeled files to create a dataset that will be sent to the hugging face models.
    :param file_de: list of german sentences
    :return: pandas df with one column called translation where each row contains a dictionary with the sentence in german
    """
    translation = []
    for input in file_de :
        trans_dict = {'de': input}
        translation.append(trans_dict)

    df = pd.DataFrame()
    df['translation'] = translation
    return df

def preprocess_function_unlabeled(samples):
    """
    This function performs preprocessing to unlabeled data before sending it to the model. This preprocessing includes
    adding a prefix of the task "translate German to English" ,tokenizing, truncating and adding padding to the data sentences.
    :param samples: dictionary with the key "translation" holding samples where each sample is a dictionary with the key "de"
    holding the data to be preprocessed.
    :return: preprocessed dataset, including original data, tokenized inputs and attention masks.
    """
    prefix = "translate German to English "

    max_input_length = 320
    source_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in samples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)
    return model_inputs


def create_translation_df_unlabeled_with_roots(file_de,roots,modifiers):
    """
    Gets a list of sentences in German , roots of their translation in english and list of roots' modifiers.
    composing out of the lists new sentences where each sentence is composed of a prefix of the roots+modifiers and then the german sentence.
    saves the new sentences to pandas df.
    :param file_de: list of sentences in german
    :param roots: list of words which are the roots of the english translation of the sentences in file_de
    :param modifiers: list of tuples of modifiers corresponding to the roots in roots list.
    :return: pandas df with one column called translation where each row contains a dictionary with the sentence in german with a prefix of the
     English roots and modifiers.
    """
    translation = []

    for input,root_str,modifiers_str in zip(file_de,roots,modifiers) :
        root_list = root_str.split(",")
        mods_tuples_list = modifiers_str.split(", (")
        mods_tuples_list = [(tup.strip("(").strip(")").strip().split(", ")) for tup in mods_tuples_list]
        prefix = ""
        for i, (root, mods) in enumerate(zip(root_list, mods_tuples_list)):
            prefix += f"ROOT_{root.strip()} "
            for j, mod in enumerate(mods):
                prefix += f"MOD{j + 1}_{mod.strip()} "
        prefix += ": "
        input = prefix + input
        raw = {'de': input}
        translation.append(raw)

    df = pd.DataFrame()
    df['translation'] = translation
    return df


def perform_inference_with_roots(file_path,model_checkpoint):
    """
    given a pretrained model and unlabeled file, performs inference and generates english translation of the germans inputs from the file
    :param file_path: path of unlabeled file on which you want to generate translation
    :param model_checkpoint: path to the pretrained model to load
    :return: 2 lists one of the german inputs and one of the generated translation in english.
    """
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
