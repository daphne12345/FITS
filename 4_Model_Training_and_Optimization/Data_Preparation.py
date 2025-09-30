import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import torch


def prepare_data():
    """
    Preprocesses the labeled data.
    :return: all data, train data, test data
    """
    print('reading data...')
    df = pd.read_csv('../data/df_annotated.csv')

    test_split = int(df.shape[0] * 0.7)
    df_train = df[:test_split]
    df_test = df[test_split:]
    
    val_split = int(df_train.shape[0] * 0.8)
    df_val = df_train[val_split:]
    df_train = df_train[:val_split]
    return df, df_train, df_val, df_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def tokenize(df, df_train, df_val, df_test, name_tokenizer):
    """
    Tokenizes the data and splits the data into training, validation and test data.
    :param df: all data
    :param df_train: the train data
    :param df_val: the validation data
    :param df_test: the test data
    :param name_tokenizer: model name
    :return: tokenizer, data_collator, train_dataset, val_dataset, test_dataset, id2label, label2id
    """
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer, truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = pd.Series(df.iloc[:, 1:].columns).to_dict()
    label2id = {v: k for k, v in id2label.items()}

    train_embeddings = tokenizer(df_train["windowed_3"].to_list(), truncation=True, padding=True)
    val_embeddings = tokenizer(df_val["windowed_3"].to_list(), truncation=True, padding=True)
    test_embeddings = tokenizer(df_test["windowed_3"].to_list(), truncation=True, padding=True)

    train_labels = df_train.iloc[:, 1:].to_numpy().astype(int).tolist()
    val_labels = df_val.iloc[:, 1:].to_numpy().astype(int).tolist()
    test_labels = df_test.iloc[:, 1:].to_numpy().astype(int).tolist()

    train_dataset = Dataset(train_embeddings, train_labels)
    val_dataset = Dataset(val_embeddings, val_labels)
    test_dataset = Dataset(test_embeddings, test_labels)
    return tokenizer, data_collator, train_dataset, val_dataset, test_dataset, id2label, label2id
