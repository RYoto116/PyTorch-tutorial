import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.], [2.], [3.]])
y_data = torch.Tensor([[0], [0], [1]])

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return F.sigmoid(self.linear(x))

model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch}, loss = {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Predict result: P(class=1) = {model(torch.Tensor([4.0])).item()}')

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()