# kuzu.py
# ZZEN9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.input = nn.Linear(28*28,10)
      

    def forward(self, x):
        out = x.view(x.size(0),-1)
        out=F.log_softmax(self.input(out),dim=1)
        return out 

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.input = nn.Linear(28 * 28, 390)
        self.output = nn.Linear(390, 10)
        # INSERT CODE HERE

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = torch.tanh(self.input(out))
        out = F.log_softmax(self.output(out),dim=1)
        return out

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.lin1   = nn.Linear(784,392)
        self.lin2   = nn.Linear(392, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1(out))
        out = F.log_softmax(self.lin2(out),dim=1)
        return out
