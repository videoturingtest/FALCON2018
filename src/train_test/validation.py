"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/validation.py
 - Contain validation code for execution for model.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch
import torch.nn.functional as F

import time


def validation(net, val_loader):
    # set testing mode
    net.eval()

    correct = 0
    total = 0
    inference_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = net(inputs.cuda())
            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
    inference_time = time.time() - inference_start
    accuracy = float(100) * float(correct) / float(total)
    print("*************** Validation ***************")
    print('Accuracy of the network validation images: %f %%' % accuracy)
    print('Validation time is: %fs' % inference_time)

    return accuracy
