import torch
import torch.nn as nn

x = torch.Tensor([[1.], [2.], [3.]])
y = torch.Tensor([[3.], [5.], [7.]])

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear =  nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

loss = None
for epoch in range(200):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, loss={loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'w={model.linear.weight.item()}, b={model.linear.bias.item()}, loss={loss.data}')
x_test = torch.Tensor([4.0])
y_pred = model(x_test)
print(y_pred.data.item())