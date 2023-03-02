import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import gzip
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
hidden_size = 100
n_layers = 2
n_epochs = 100
n_chars = 128
n_epochs = 30

def create_tensor(tensor):
    return tensor.to(device)

# 将一个名字转为字符列表
def name2list(name):
    charlist = [ord(c) for c in name] # 使用字符的ascii编码，input_size=128
    return charlist, len(charlist)

def make_tensor(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()
    batchSize = len(name_sequences)
    seqLen = seq_lengths.max()

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(batchSize, seqLen).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

# Preparing Data
class NameDataset(Dataset):
    def __init__(self, is_train_set):
        self.file_name = '../data/names/names_train.csv.gz' if is_train_set else '../data/names/names_test.csv.gz'
        with gzip.open(self.file_name, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.names = [row[0] for row in rows]
            self.len = len(self.names)
            self.countries = [row[1] for row in rows]
            self.country_list = list(sorted(set(self.countries))) # 去重
            self.country_dict = self.getCountryDict()
            self.country_num = len(self.country_dict)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country in enumerate(self.country_list, 0):
            country_dict[country] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num

train_set = NameDataset(is_train_set=True)
test_set = NameDataset(is_train_set=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# N_COUNTRY is the output size of our model.
n_countries = train_set.getCountriesNum()

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def forward(self, inputs, seq_lengths):
        inputs = inputs.t()
        batch_size = inputs.size(1)
        embedding = self.embedding(inputs) # seqlen * batchsize * hiddensize
        hidden = self._init_hidden(batch_size)

        gru_input = pack_padded_sequence(embedding, seq_lengths.cpu())
        output, hidden = self.rnn(gru_input, hidden)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        return self.fc(hidden_cat)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_directions * n_layers, batch_size, hidden_size)
        return create_tensor(hidden)

model = RNNClassifier(n_chars, hidden_size, n_countries, n_layers, bidirectional=True)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def trainModel():
    total_loss = 0.0
    for i, (name, country) in enumerate(train_loader, 1):
        inputs, seq_lengths, target = make_tensor(name, country)
        output = model(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(train_set)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss

def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model ...")

    with torch.no_grad():
        for i, (name, country) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = make_tensor(name, country)
            output = model(inputs, seq_lengths)
            _, pred = output.max(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    start = time.time()
    print("Training for %d epochs..." % n_epochs)
    acc_list = []
    for epoch in range(n_epochs):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    epochs = np.arange(0, n_epochs)
    plt.plot(epochs, acc_list)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()