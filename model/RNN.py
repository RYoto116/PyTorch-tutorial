import torch
import torch.nn as nn

num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2] # batch_size * seq_len

# x_one_hot = torch.nn.functional.one_hot(torch.LongTensor(x_data), num_classes=num_class)
# inputs = x_one_hot.view(batch_size, -1, input_size)
# labels = torch.LongTensor(y_data).view(-1, 1) # seqlen * batchsize

inputs = torch.LongTensor(x_data).view(batch_size, -1).to(device)
labels = torch.LongTensor(y_data).to(device)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class) # seqlen * batchsize, num_class

model = RNNModel()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.cpu().data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/30] loss = %.3f' % (epoch + 1, loss.item()))
