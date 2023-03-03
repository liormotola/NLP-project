import torch.cuda

from preprocess import create_train_df, T5DataSet
from train import train




if __name__ == '__main__':
    with open("data/train.labeled", "r", encoding='utf8') as train_file:
        train_text = train_file.read()
    train_df = create_train_df(train_text)
    train_dataset = T5DataSet(train_df)

    with open("data/val.labeled", "r", encoding='utf8') as test_file:
        test_text = test_file.read()
    test_df = create_train_df(test_text)
    test_dataset = T5DataSet(test_df)
    train(train_dataset, test_dataset, batch_size=16)

