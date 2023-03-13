import pandas as pd
from project_evaluate import read_file
import random
import spacy


def create_data_with_roots(file):
    """
    Gets a text file containing sentences in German and their translation in English.
    Parsing the file + for each sentence in english finds its root and 2 of its modifiers if exist
    saves all data to a pandas df + csv called "train_data_with_roots.csv"
    :param file: path to *labeled* file to parse
    :return: pandas data frame containing 4 columns - sentence in german, translation in english, roots in english, root's modifiers in english
    """
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


def create_translation_df_with_roots(df):
    """
    preparing data to create a dataset that will be sent to the hugging face models.
    :param df: pandas df containing the data : german+english sentences , root and modifiers in english.
    :return: pandas df with one column called translation where each row contains a dictionary with the sentence in german with a prefix of the
     English roots and modifiers , and translation in English.
    """
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
        trans_dict = {'de': input, 'en': target}
        translation.append(trans_dict)

    translation_df = pd.DataFrame()
    translation_df['translation'] = translation
    return translation_df



def create_translation_df_val_with_roots(file_de,roots,modifiers,file_en):
    """
    preparing validation data to create a dataset that will be sent to the hugging face models.
    The validation data already has information about roots and modifiers therefor requires different treatment.

    :param file_de: list of sentences in german
    :param roots: list of roots per sample in english
    :param modifiers: list of tuples containing up to 2 modifiers for each root in roots argument
    :param file_en: list of sentences which are the corresponding english translation of the german list.
    :return: pandas df with one column called translation where each row contains a dictionary with the sentence in german with a prefix of the
     English roots and modifiers , and translation in English.
    """
    translation = []

    for input,root_str,modifiers_str,target in zip(file_de,roots,modifiers,file_en) :
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
        trans_dict = {'de': input,'en': target}
        translation.append(trans_dict)

    translation_df = pd.DataFrame()
    translation_df['translation'] = translation
    return translation_df


def read_file_unlabeled_with_roots(file_path):
    """
    given a file in unlabeled format, returns list of the german sentences, list of roots in english and list of the root's
    modifiers
    :param file_path: path to the file to parse
    :return: 3 lists of the parsed data: sentences in german, roots, modifiers
    """
    file_de = []
    roots = []
    modifiers = []

    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line[:17] == "Roots in English:" :
                roots.append(line[18:])
                continue
            if line[:21] == "Modifiers in English:":
                modifiers.append(line[22:])
                continue
            if line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                cur_list = file_de
                continue
            cur_str += line + ' '

    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_de, roots, modifiers

