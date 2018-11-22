"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/test.py
 - test the pre-trained model on test datasets.
 - print the test accuracy and inference time.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch
import torch.nn.functional as F

import time

from utils.load_data import load_cifar10
from utils.load_data import load_cifar100


def test(net, batch_size=128, data='cifar100'):
    # set testing mode
    net.eval()
    is_train = False

    # data
    if data == 'cifar10':
        test_loader = load_cifar10(is_train, batch_size)
    elif data == 'cifar100':
        test_loader = load_cifar100(is_train, batch_size)
    else:
        pass

    correct = 0
    total = 0
    inference_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = net(inputs.cuda())
            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
    inference_time = time.time() - inference_start
    print('Accuracy of the network on the 10000 test images: %f %%' % (float(100) * float(correct) / float(total)))
    print('Inference time is: %fs' % inference_time)

    return inference_time
