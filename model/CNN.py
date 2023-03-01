import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])
train_dataset = datasets.MNIST(root='../data/mnist/', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='../data/mnist/', train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2) # 无权重，只需要一个实例
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

cnn = CNNModel()
cnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

def train(epoch):
    running_loss = 0.0
    for idx, (x, y) in enumerate(train_loader, 0):
        x = x.to(device)
        y = y.to(device)
        outputs = cnn(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 300))
            running_loss = 0.0

def modelTest():
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for(x, y) in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = cnn(x)
            _, pred = torch.max(outputs.data, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 3 == 0:
            modelTest()