"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: model/simplenet_standardConv.py
 - Contain a simple-net class.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch.nn as nn


class SimpleNetOriginal(nn.Module):
    """
    Description: simplenet with standard convolution class.
    """
    def __init__(self, num_classes=10):
        """
        Initialize SimpleNet with standard convolution as argument configurations.
        :param num_classes: number of classification labels
        """
        super(SimpleNetOriginal, self).__init__()

        # standard conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2)
        )

        # standard conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2)
        )

        # standard fc layer
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

