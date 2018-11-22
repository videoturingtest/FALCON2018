"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/save_restore.py
 - Contain source code for saving and restoring the model.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import os
import torch


def save_model(best, args):
    """
    Save the trained model.
    :param best: best trained model (to be saved)
    :param args: arguments of the trained model
    """
    name = 'conv=' + str(args.convolution) + \
           ',model=' + str(args.model) + \
           ',data=' + str(args.datasets) + \
           ',rank=' + str(args.rank) + \
           ',alpha=' + str(args.alpha)
    path = 'trained_model/'
    mkdir(path)
    torch.save(best, path + name + '.pkl')


def load_model(net, args):
    """
    Restore the pre-trained model.
    :param net: model architecture without parameters
    :param args: arguments of the trained model
    """
    name = 'conv=' + str(args.convolution) + \
           ',model=' + str(args.model) + \
           ',data=' + str(args.datasets) + \
           ',rank=' + str(args.rank) + \
           ',alpha=' + str(args.alpha)

    path = 'trained_model/'
    net.load_state_dict(torch.load(path + name + '.pkl'))


def mkdir(path):
    """
    Make a directory if it doesn't exist.
    :param path: directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
