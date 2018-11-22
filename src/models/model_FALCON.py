"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: model/model_FALCON.py
 - Contain model with FALCON class.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch.nn as nn

from .simplenet_FALCON import BlockOutputEHP


class FALCONModel(nn.Module):
    """
    Description: FALCON Model class.
    """

    # configures of different models
    cfgs_VGG16 = [(3, 64), (64, 64, 2),
            (64, 128), (128, 128, 2),
            (128, 256), (256, 256), (256, 256, 2),
            (256, 512), (512, 512), (512, 512, 2),
            (512, 512), (512, 512), (512, 512, 2)]

    cfgs_VGG19 = [(3, 64), (64, 64, 2),
              (64, 128), (128, 128, 2),
              (128, 256), (256, 256), (256, 256), (256, 256, 2),
              (256, 512), (512, 512), (512, 512), (512, 512, 2),
              (512, 512), (512, 512), (512, 512), (512, 512, 2)]

    cfgs_MobileNet = [(32, 64), (64, 128, 2),
              (128, 128), (128, 256, 2),
              (256, 256), (256, 512, 2),
              (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 1024, 2),
              (1024, 1024, 2)]

    def __init__(self, num_classes=10, which='VGG16'):
        """
        Initialize FALCON Model as argument configurations.
        :param num_classes: number of classification labels
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        """

        super(FALCONModel, self).__init__()

        if which == 'MobileNet':
            self.conv = nn.Sequential(
                nn.Conv2d(3,
                          32,
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential()

        self.layers = self._make_layers(which)

        self.avgPooling = nn.AvgPool2d(2, 2)
        if which == 'MobileNet':
            output_size = 1024
        else:
            output_size = 512
        self.fc = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, which):
        """"
        Make FALCON Model layers.
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        """
        layers = []
        if which == 'VGG16':
            self.cfgs = self.cfgs_VGG16
        elif which == 'VGG19':
            self.cfgs = self.cfgs_VGG19
        elif which == 'MobileNet':
            self.cfgs = self.cfgs_MobileNet
        else:
            pass

        for cfg in self.cfgs:
            if len(cfg) == 3:
                stride = cfg[2]
            else:
                stride = 1
            layers.append(BlockOutputEHP(cfg[0], cfg[1], stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        # out = self.avgPooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
