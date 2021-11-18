import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, in_channel, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_shape = 16 * 4 * 4
        self.fc1 = nn.Linear(self.fc_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):

    def __init__(self, n_class):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, n_class)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN1(nn.Module):

    def __init__(self, in_channel, n_class):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # reshape for both MNIST and CIFAR based on # of channels
        self.fc_shape = 16 * int(4.5 + in_channel * 0.5) * int(4.5 + in_channel * 0.5)
        self.fc1 = nn.Linear(self.fc_shape, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN2(nn.Module):

    def __init__(self, in_channel, n_class):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.fc_shape = 128 * int(4.5 + in_channel * 0.5) * int(4.5 + in_channel * 0.5)
        self.fc = nn.Linear(self.fc_shape, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = self.fc(x)
        return x
