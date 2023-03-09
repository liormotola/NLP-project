import pandas as pd

if __name__ == '__main__':
    english_file = "/home/student/Final Project/Lior/new_data_english_generated_by_t5_model_trained_for_10_ep.txt"
    german_file = "/home/student/Final Project/Lior/new_data_german_t5_trained_10_ep.txt"

    with open(german_file,"r",encoding="utf-8") as f:
        german_lines = f.readlines()
        german_lines = [line.strip() for line in german_lines if line.strip()]

    with open(english_file,"r",encoding="utf-8") as f:
        english_lines = f.readlines()
        english_lines = [line.strip()for line in english_lines if line.strip()]

    data = {"German":[], "English":[]}
    data["German"] = german_lines
    data["English"] = english_lines
    df = pd.DataFrame(data)
    df.to_csv("generated_ger_en.csv",index=False)

    # translation = []
    # for input, target in zip(german_lines[:1000], english_lines[:1000]):
    #     raw = {'de': input, 'en': target}
    #     translation.append(raw)
    #
    # dataset = pd.DataFrame()
    # dataset['translation'] = translation
    #
    # dataset.to_csv("ger-en_new_data_generated_by_t5.csv",index=False)
