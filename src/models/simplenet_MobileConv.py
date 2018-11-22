"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: model/simplenet_MobileConv.py
 - Contain Mobile-conv class and a simple-net with Mobile-conv class.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch.nn as nn


class MobileConv(nn.Module):
    """
    Description: Mobile-conv class.
    """
    def __init__(self, in_planes, out_planes, stride=1):
        """
        Initialize Mobile-conv as argument configurations.
        :param in_planes: number of input channel
        :param out_planes: number of output channle
        :param stride: stride size
        """

        super(MobileConv, self).__init__()

        # Mobile-conv includes BN and ReLU after both dw and pw layer
        self.Block = nn.Sequential(
            nn.Conv2d(in_planes, in_planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.Block(x)


class SimpleNet_Mobile(nn.Module):
    """
    Description: SimpleNet (with Mobile-conv) class.
    """
    def __init__(self, num_classes=10):
        """
        Initialize SimpleNet with Mobile-conv as argument configurations.
        :param num_classes: number of classification labels
        """
        super(SimpleNet_Mobile, self).__init__()

        # standard conv layer replaced by Mobile-conv
        self.conv1 = MobileConv(3, 64, 2)
        self.conv2 = MobileConv(64, 64, 2)

        # fc layer remains unchanged
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 384),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(192, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
