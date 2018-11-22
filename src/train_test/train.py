"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/train_test.py
 - Contain training code for execution for model.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

from train_test.validation import validation
from utils.optimizer_option import get_optimizer
from utils.load_data import load_cifar10
from utils.load_data import load_cifar100
from utils.lr_decay import adjust_lr


def train(net,
          lr,
          optimizer_option='SGD',
          data='cifar100',
          epochs=350,
          batch_size=128,
          n_lr_decay=False,
          is_train=True):

    net.train()

    if data == 'cifar10':
        trainloader, valloader = load_cifar10(is_train, batch_size)
    elif data == 'cifar100':
        trainloader, valloader = load_cifar100(is_train, batch_size)
    else:
        exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, lr, optimizer_option)

    start_time = time.time()
    last_time = 0

    best_acc = 0
    best_param = net.state_dict()

    iteration = 0
    for epoch in range(epochs):
        print("****************** EPOCH = %d ******************" % epoch)

        total = 0
        correct = 0
        loss_sum = 0

        # change learning rate
        if not n_lr_decay:
            if epoch % 70 == 69:
                lr = adjust_lr(lr, epoch)
                optimizer = get_optimizer(net, lr, optimizer_option)

        for i, data in enumerate(trainloader, 0):
            iteration += 1

            # foward
            inputs, labels = data
            inputs_V, labels_V = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs_V)
            loss = criterion(outputs, labels_V)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels_V.size(0)
            correct += (predicted == labels_V).sum()
            loss_sum += loss

            if iteration % 100 == 99:
                now_time = time.time()
                print('accuracy: %f %%; loss: %f; time: %ds'
                      % ((float(100) * float(correct) / float(total)), loss, (now_time - last_time)))
                total = 0
                correct = 0
                loss_sum = 0
                last_time = now_time

        # validation
        if epoch % 5 == 4:
            net.eval()
            val_acc = validation(net, valloader)
            net.train()
            if val_acc > best_acc:
                best_acc = val_acc
                best_param = net.state_dict()

    print('Finished Training. It took %ds in total' % (time.time() - start_time))
    return best_param
