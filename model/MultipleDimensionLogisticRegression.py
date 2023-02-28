import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

xy = np.loadtxt('C:/Users/Administrator/Downloads/PyTorch Tutorials/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

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

class ReLUModel(nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = nn.functional.sigmoid(self.linear3(x))
        return x

logisticModel = Model()
reluModel = ReLUModel()

criterion = nn.BCELoss()
logisticOptimizer = torch.optim.Adam(logisticModel.parameters(), lr=0.1)
reluOptimizer = torch.optim.Adam(reluModel.parameters(), lr=0.1)
logitic_list = []
relu_list = []

for epoch in range(100):
    y_pred = logisticModel(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch}, loss = {loss:.2f}')
    logitic_list.append(loss.item())
    logisticOptimizer.zero_grad()
    loss.backward()
    logisticOptimizer.step()

for epoch in range(100):
    y_pred = reluModel(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch}, loss = {loss:.2f}')
    relu_list.append(loss.item())
    reluOptimizer.zero_grad()
    loss.backward()
    reluOptimizer.step()

epochs = np.arange(0, 100)
plt.plot(epochs, logitic_list, color='blue', label='logistic function')
plt.plot(epochs, relu_list, color='orange', label='relu funciton')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()