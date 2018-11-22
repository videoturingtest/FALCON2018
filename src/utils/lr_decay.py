"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/lr_decay.py
 - Contain source code for updating learning rate.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact tshe authors.

"""


def adjust_lr(lr):
    """
    Update learnign rate.
    :param lr: original learning rate.
    """
    lr = lr / 10
    print("learning rate change to %f" % lr)
    return lr

