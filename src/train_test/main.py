"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/train_test.py
 - receive arguments and train_test/test the model.

Version: 1.0

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

"""

import sys
sys.path.append('../')

from train_test.train import train
from train_test.test import test

from models.model_standardConv import StandardConvModel
from models.model_MobileConv import MobileConvModel
from models.model_FALCON import FALCONModel
from models.model_MobileConv_rank import RankMobileConvModel
from models.model_FALCON_rank import RankFALCONModel

from utils.default_param import get_default_param
from utils.save_restore import save_model
from utils.save_restore import load_model
from utils.compression_cal import cr_crr


def main(args):

    if args.datasets == "cifar10":
        num_classes = 10
    elif args.datasets == "cifar100":
        num_classes = 100
    # elif args.datasets == "imagenet":
    #     num_classes = 1000
    else:
        pass

    if args.convolution == "MobileConv":
        net = MobileConvModel(num_classes=num_classes, which=args.model)
    elif args.convolution == "FALCON":
        net = FALCONModel(num_classes=num_classes, which=args.model)
    elif args.convolution == "RankMobileConv":
        net = RankMobileConvModel(rank=args.rank, alpha=args.alpha, num_classes=num_classes, which=args.model)
    elif args.convolution == "RankFALCON":
        net = RankFALCONModel(rank=args.rank, alpha=args.alpha, num_classes=num_classes, which=args.model)
    elif args.convolution == "StandardConv":
        net = StandardConvModel(num_classes=num_classes, which=args.model)
    else:
        pass

    net = net.cuda()

    if args.is_train:
        # training
        best = train(net,
                     lr=args.learning_rate,
                     optimizer_option=args.optimizer,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     n_lr_decay=args.no_learning_rate_dacay,
                     is_train=args.is_train,
                     data=args.datasets)
        save_model(best, args)
        test(net, batch_size=args.batch_size, data=args.datasets)
        cr_crr(args)
    else:
        # testing
        load_model(net, args)
        inference_time = 0
        for i in range(10):
            inference_time += test(net, batch_size=args.batch_size, data=args.datasets)
        print("Averate Inference Time: %fs" % (float(inference_time) / float(10)))
        # cr_crr(args)


if __name__ == "__main__":
    parser = get_default_param()
    args = parser.parse_args()
    print(args)

    main(args)


