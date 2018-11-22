"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/load_data.py
 - Contain source code for loading data.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact tshe authors.

"""

import torch
import torchvision
import torchvision.transforms as transforms


# CIFAR10 data
def load_cifar10(is_train=True, batch_size=128):
    """
    Load cifar-10 datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if is_train:
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='./data',
                                              train=False,
                                              download=True,
                                              transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2
                                                )
        return trainloader, valloader

    else:
        testset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader


# CIFAR100 data
def load_cifar100(is_train=True, batch_size=128):
    """
    Load cifar-100 datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if is_train:
        trainset = torchvision.datasets.CIFAR100(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)
        return trainloader, valloader

    else:
        testset = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader
