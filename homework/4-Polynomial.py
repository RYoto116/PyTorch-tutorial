import numpy as np
import  torch
import matplotlib.pyplot as plt

x_data = np.array([1., 2., 3.])
y_data = np.array([3., 5., 7.])

w = torch.randn(2,)
b = torch.randn(1,)
w.requires_grad = True
b.requires_grad = True

x_data = np.append([x_data], [x_data ** 2], axis=0).transpose()

def forward(x):
    return torch.sum(w * x) + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

l = None
alpha = 0.01
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        x = torch.tensor(x)
        l = loss(x, y)
        l.backward()
        print(f'data: ({x.data}, {y.data}), grad: {w.grad.data}')
        w.data -= alpha * w.grad.data
        b.data -= alpha * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
    print(f'Epoch: {epoch}, loss = {l.item()}')
    loss_list.append(l.item())

print(f'predict: 4, {forward(4).item()}')

epochs = np.arange(0, 100)
plt.plot(epochs, loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
