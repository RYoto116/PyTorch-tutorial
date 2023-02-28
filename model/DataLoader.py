import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):
    def __init__(self):
        super(DiabetesDataset, self).__init__()
        xy = np.loadtxt('C:/Users/Administrator/Downloads/PyTorch Tutorials/diabetes.csv.gz', delimiter=',',
                        dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_list = []
num_epoch = 20

if __name__ == '__main__':
    for epoch in range(num_epoch):
        batch_losses = []
        for i, (x, y) in enumerate(train_loader, 0):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            batch_losses.append(loss.item())
            print(f'Epoch: {epoch}, Iteration: {i}, loss={loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(np.mean(batch_losses))

    plt.plot(np.arange(0, num_epoch, 1), loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()