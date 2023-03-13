import pandas as pd


def create_translation_df(german,english):
    """
    preparing data to create a dataset that will be sent to the hugging face models.

    :param german: list of sentences in german
    :param english: list of sentences in english corresponding to the one in the german list
    :return: pandas df with one column called translation where each row contains a dictionary with the sentence in german and its translation in English.
    """
    translation = []
    for input, target in zip(german, english):
        trans_dict = {'de': input, 'en': target}
        translation.append(trans_dict)

    df = pd.DataFrame()
    df['translation'] = translation
    return df

def postprocess_text(preds, labels):
    """
    This function is used to compute metrics while training, postprocess the results by using strip and lower().
    :param preds: models predictions (generated sentences)
    :param labels: true labels
    :return: postprocessed preds and labels
    """
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]

    return preds, labels



