"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/default_param.py
 - Contain source code for receiving arguments .

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import argparse


def get_default_param():
    """
    Receive arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-train", "--is_train",
                        help="whether train_test the model (train_test-True; test-False)",
                        action="store_true")

    parser.add_argument("-bs", "--batch_size",
                        help="batch size of training",
                        type=int,
                        default=128)

    parser.add_argument("-ep", "--epochs",
                        help="epochs of training",
                        type=int,
                        default=200)

    parser.add_argument("-lr", "--learning_rate",
                        help="set beginning learning rate",
                        type=float,
                        default=1e-2)

    parser.add_argument("-op", "--optimizer",
                        help="choose optimizer",
                        choices=["SGD", "Adagrad", "Adam", "RMSprop"],
                        type=str,
                        default='SGD')

    parser.add_argument("-n_lr_dec", "--no_learning_rate_dacay",
                        help="whether lr will change(decrease) during training (True/False)",
                        action="store_false")

    parser.add_argument("-conv", "--convolution",
                        help="choose model",
                        choices=["StandardConv",
                                 "MobileConv",
                                 "FALCON",
                                 "RankMobileConv",
                                 "RankFALCON"],
                        type=str,
                        default="FALCON")

    parser.add_argument("-k", "--rank",
                        help="if the model is Rank K, the rank(k) in range {2,3,4}",
                        choices=[1, 2, 3, 4],
                        type=int,
                        default=1)

    parser.add_argument("-al", "--alpha",
                        help="Width Multiplier in range (0,1]",
                        # choices=[1, 0.75, 0.5, 0.33, 0.25],
                        type=float,
                        default=1)

    parser.add_argument("-m", "--model",
                        help="model type - VGG16/VGG19/MobileNet",
                        choices=['VGG16', 'VGG19', 'MobileNet'],
                        type=str,
                        default='MobileNet')

    parser.add_argument("-data", "--datasets",
                        help="specify datasets - cifar10/cifar100/imagenet",
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        type=str,
                        default='cifar100')

    return parser
