import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [3., 5., 7.]

w = torch.Tensor([1.])
b = torch.Tensor([0.])
w.requires_grad = True
b.requires_grad = True

def forward(x):
    return w * x + b # 重载为张量乘法，x自动转换为tensor

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
# 每次调用loss函数就动态构建一个计算图

l = None
alpha = 0.1
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() # 梯度存入张量之后，自动释放计算图
        print(f'data: ({x}, {y}), grad: {w.grad.item()}')
        w.data -= alpha * w.grad.data # 取张量的data进行计算，不会建立计算图
        b.data -= alpha * b.grad.data
        w.grad.data.zero_() # 将参数的梯度数据全部清零
        b.grad.data.zero_()
    print(f'Epoch: {epoch}, loss = {l.item()}')
    loss_list.append(l.item())

print(f'predict: 4, {forward(4).item()}')

epochs = np.arange(0, 100)
plt.plot(epochs, loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
