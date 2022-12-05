import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.conv3 = nn.Conv2d(20,40, 3, 1)
        self.fc1 = nn.Linear(17*17*40, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60,25)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 17*17*40)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)