from preprocess import create_train_df





if __name__ == '__main__':
    with open("data/train.labeled", "r", encoding='utf8') as train_file:
        train_text = train_file.read()
    df = create_train_df(train_text)
    roni=5