# 随机梯度下降：用单个样本的损失函数作为参数更新的依据
# 随机梯度下降计算单个样本损失的同时更新参数，更新后的参数用于计算下一个样本的损失，因此不能并行计算
# 时间复杂度：O(epoch * N)
# 使用SGD的主要原因是避免陷入鞍点

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0

def forward(x):
    return w * x

def cost(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

loss = None
alpha = 0.1
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss = cost(x, y)
        grad = gradient(x, y)
        w -= alpha * grad
    print(f'Epoch: {epoch}, w={w}, loss={loss}')
print(f'Final parameters: w={w}, Loss={loss}')