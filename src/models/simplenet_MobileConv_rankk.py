"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: model/simplenet_MobileConv_rankk.py
 - Contain rank-k Mobile-conv class and a simple-net with rank-k Mobile-conv class.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch.nn as nn


class MobileConvRank(nn.Module):
    """
    Description: rank-k Mobile-conv class.
    """
    def __init__(self, in_planes, out_planes, stride=1, rank=2):
        """
        Initialize rank-k Mobile-conv as argument configurations.
        :param in_planes: number of input channel
        :param out_planes: number of output channle
        :param stride: stride size
        :param rank: rank of convolution -
                     copy the conv layer for n times,
                     run independently
                     and add output together at the end of the layer
        """

        super(MobileConvRank, self).__init__()
        self.rank = rank

        for i in range(self.rank):
            setattr(self, 'Block'+str(i),
                    nn.Sequential(
                        nn.Conv2d(in_channels=in_planes,
                                  out_channels=in_planes,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  groups=in_planes),
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels=in_planes,
                                  out_channels=out_planes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
                        )
                    )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for i in range(self.rank):
            if i == 0:
                out = getattr(self, 'Block'+str(i))(x)
            else:
                out += getattr(self, 'Block'+str(i))(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SimpleNetInputEHPRankK(nn.Module):
    """
    Description: SimpleNet with rank-k Mobile-conv class.
    """
    def __init__(self, rank=2, num_classes=10):
        """
        Initialize SimpleNet with rank-k Mobile-conv as argument configurations.
        :param num_classes: number of classification labels
        :param rank: rank of convolution
        """
        super(SimpleNetInputEHPRankK, self).__init__()

        # standard conv layer replaced by rank-k Mobile-conv block
        self.conv1 = MobileConvRank(3, 64, 2, rank=rank)
        self.conv2 = MobileConvRank(64, 64, 2, rank=rank)

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
