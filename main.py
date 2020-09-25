
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from keras.preprocessing.text import Tokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import pickle
import itertools
from statistics import mean

from dataloader_ import Dataloader_
from model import Model

class Params(object):
    dataset_dir = "data/4900_news.xlsx"
    test_split_rate = 0.3
    label_tokenizer = None
    shuffle_count = 50
    batch_size = 10 # 64
    batch_shuffle = True
    bert_out_features_size = 768
    hidden_1_size = 128
    hidden_2_size = 64
    lr = 2e-5 # 0.025
    epoch = 1 # 200
    bert_sequence_max_length = 100
    model = None
    optimizer = None
    criterion = None

    use_cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_tokenizer_name = "label_tokenizer.pickle"

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    bert_tokenizer_dir = os.path.join("model", "bert", "tokenizer")
    if not os.path.exists(bert_tokenizer_dir):
        os.makedirs(bert_tokenizer_dir)

    bert_models_dir = os.path.join("model", "bert", "model")
    if not os.path.exists(bert_models_dir):
        os.makedirs(bert_models_dir)

    # download bert models
    bert_model_name = "loodos/albert-base-turkish-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    for param in bert_model.parameters(): # freeze bert model params
        param.requires_grad = False

    # save downloaded tokenizer / models
    bert_tokenizer.save_pretrained(bert_tokenizer_dir)
    bert_model.save_pretrained(bert_models_dir)

class Main(object):

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        data = pd.read_excel(Params.dataset_dir)
        return data

    @staticmethod
    def run_preprocess():
        data = Main.load_dataset()

        """
        # preprocess
        data = lowercase(data)
        data = remove_punctuations(data)
        data = remove_numbers(data)
        data = remove_stop_words(data)
        # data = zemberek_stemming(data)
        data = first_5_char_stemming(data)
        data = data_shuffle(data, Params.shuffle_count)
        """

        X_train, X_test, y_train, y_test = train_test_split(data["text"],
                                                            data["label"],
                                                            test_size=Params.test_split_rate,
                                                            stratify=data["label"],
                                                            shuffle=True,
                                                            random_state=42)

        Params.label_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        Params.label_tokenizer.fit_on_texts(y_train)
        with open(os.path.join(Params.model_dir, Params.label_tokenizer_name), "wb") as handle:
            pickle.dump(Params.label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_train = Params.label_tokenizer.texts_to_sequences(y_train)  # list of list
        y_train = [_y[0] - 1 for _y in y_train] # -1 for start from 0 index
        y_test = Params.label_tokenizer.texts_to_sequences(y_test)
        y_test = [_y[0] - 1 for _y in y_test]  # [batch]

        # convert series to list
        X_train = X_train.tolist()
        X_test = X_test.tolist()

        # create dataloaders
        train_dataloader = DataLoader(dataset=Dataloader_(X_train, y_train),
                                      batch_size=Params.batch_size,
                                      shuffle=Params.batch_shuffle)
        test_dataloader = DataLoader(dataset=Dataloader_(X_test, y_test),
                                     batch_size=Params.batch_size,
                                     shuffle=Params.batch_shuffle)

        return train_dataloader, test_dataloader

    @staticmethod
    def run_train(dataloader):
        Params.model.train()  # set train mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            print("train batch id: ", id)
            encoding = Params.bert_tokenizer(list(X),
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=Params.bert_sequence_max_length,
                                             add_special_tokens=True)
            #print("encoding:", encoding)
            input_ids = encoding['input_ids']
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding['attention_mask']

            # convert tensors to variables
            input_ids = Variable(input_ids, requires_grad=False)
            token_type_ids = Variable(token_type_ids, requires_grad=False)
            attention_mask = Variable(attention_mask, requires_grad=False)
            y = Variable(y, requires_grad=False)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            outputs = Params.model(input_ids, token_type_ids, attention_mask)
            #print(outputs)
            loss = Params.criterion(outputs, y.long())  # y: long type
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

            # backward and optimize
            Params.optimizer.zero_grad()
            loss.backward()
            Params.optimizer.step()

            torch.cuda.empty_cache()

        pred_flat = list(itertools.chain(*predicted_labels)) # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), acc

    @staticmethod
    def run_test(dataloader):
        Params.model.eval()  # set eval mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, (X, y) in enumerate(dataloader):  # batch
            encoding = Params.bert_tokenizer(list(X),
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=Params.bert_sequence_max_length,
                                             add_special_tokens=True)
            # print("encoding:", encoding)
            input_ids = encoding['input_ids']
            token_type_ids = encoding["token_type_ids"]
            attention_mask = encoding['attention_mask']

            # convert tensors to variables
            input_ids = Variable(input_ids, requires_grad=False)
            token_type_ids = Variable(token_type_ids, requires_grad=False)
            attention_mask = Variable(attention_mask, requires_grad=False)
            y = Variable(y, requires_grad=False)

            if Params.use_cuda:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                y = y.cuda()

            outputs = Params.model(input_ids, token_type_ids, attention_mask)
            # print(outputs)
            loss = Params.criterion(outputs, y.long())
            epoch_loss.append(loss.item())  # save batch loss

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted.tolist())
            y_labels.append(y.tolist())

            torch.cuda.empty_cache()

        pred_flat = list(itertools.chain(*predicted_labels))  # flat list of list to list
        y_flat = list(itertools.chain(*y_labels))
        acc = Main.get_metrics(pred_flat, y_flat)
        return mean(epoch_loss), acc

    @staticmethod
    def get_metrics(y_pred, y_true):
        acc = accuracy_score(y_true, y_pred)
        return acc

    @staticmethod
    def run_train_test(train_dataloader, test_dataloader):
        Params.model = Model(bert_model=Params.bert_model,
                             bert_out_features_size=Params.bert_out_features_size,
                             hidden_1_size=Params.hidden_1_size,
                             hidden_2_size=Params.hidden_2_size,
                             class_size=len(Params.label_tokenizer.word_index))
        # push model to gpu
        if Params.use_cuda:
            Params.model = Params.model.cuda()

        pytorch_total_params = sum(p.numel() for p in Params.model.parameters())
        print("pytorch_total_params: ", pytorch_total_params)

        Params.optimizer = torch.optim.Adam(Params.model.parameters(), lr=Params.lr)
        Params.criterion = nn.CrossEntropyLoss()

        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        for epoch in range(1, Params.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            _train_loss, _train_acc = Main.run_train(train_dataloader)
            train_loss.append(_train_loss)
            train_acc.append(_train_acc)

            # test
            _test_loss, _test_acc = Main.run_test(test_dataloader)
            test_loss.append(_test_loss)
            test_acc.append(_test_acc)

            # info
            print("train loss -> ", _train_loss)
            print("train acc -> ", _train_acc)

            print("test loss -> ", _test_loss)
            print("test acc -> ", _test_acc)

        torch.save(Params.model.state_dict(), os.path.join(Params.model_dir, str(epoch) + "_model.pt"))

if __name__ == '__main__':

    print("cuda available: ", torch.cuda.is_available())

    train_dataloader, test_dataloader = Main.run_preprocess()
    Main.run_train_test(train_dataloader, test_dataloader)

