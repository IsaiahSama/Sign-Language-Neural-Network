"""This file is responsible for building and training the sign language
classifier using deep learning.

This file will define a model and train it on the data.

We will build a neural network with six layers, define a loss, an optimizer and
finally, optimize the loss function for your neural network predictions.

"""

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from step_2_dataset import get_train_test_loaders


class Net(nn.Module):
    """This is a Pytorch neural network.

    It contains three convolutional layers, followed by three fully connected
    layers
    """

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3).to("cuda")
        self.pool = nn.MaxPool2d(2, 2).to("cuda")
        self.conv2 = nn.Conv2d(6, 6, 3).to("cuda")
        self.conv3 = nn.Conv2d(6, 16, 3).to("cuda")

        self.fc1 = nn.Linear(16 * 5 * 5, 120).to("cuda")
        self.fc2 = nn.Linear(120, 48).to("cuda")
        self.fc3 = nn.Linear(48, 24).to("cuda")

    def forward(self, x):
        x = F.relu(self.conv1(x)).to("cuda")
        x = self.pool(F.relu(self.conv2(x)).to("cuda"))
        x = self.pool(F.relu(self.conv3(x)).to("cuda"))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x)).to("cuda")
        x = F.relu(self.fc2(x)).to("cuda")
        x = self.fc3(x)
        return x


def train(net: Net, criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, trainloader: torch.utils.data.DataLoader, epoch: int):
    """This method will run the forward pass and then back propagate through
    the loss and neural network"""

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs = Variable(data["image"].float()).to("cuda")
        labels = Variable(data["label"].long()).to("cuda")
        optimizer.zero_grad()

        # Forward + backward + optimize

        outputs = net(inputs)
        loss = criterion(outputs, labels[:, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print("[%d, %5d] loss: %.6f" % (epoch, i, running_loss / (i + 1)))


def main():
    net = Net().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainloader, _ = get_train_test_loaders()

    # An epoch is an iteration of training where every training sample
    # has been used exactly once.
    for epoch in range(2):
        train(net, criterion, optimizer, trainloader, epoch)
        scheduler.step()

    torch.save(net.state_dict(), "checkpoint.pth")


if __name__ == "__main__":
    main()
