# 梯度下降：将所有样本的平均损失作为参数更新的依据
# 梯度下降中，用同一个参数计算所有单个样本的损失，求平均后再更新参数，单个样本可以并行计算
# 时间复杂度：O(epoch)

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0

def forward(x):
    return w * x

def cost(xs, ys): # 并行后O(1)
    loss = 0
    n_sample = len(xs)
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        loss += (y_pred - y) * (y_pred - y)
    return loss / n_sample

def gradient(xs, ys): # 并行后O(1)
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

loss = None
alpha = 0.1
for epoch in range(100):
    loss = cost(x_data, y_data)
    grad = gradient(x_data, y_data)
    w -= alpha * grad
    print(f'Epoch: {epoch}, w={w}, loss={loss}')
print(f'Final parameters: w={w}, Loss={loss}')