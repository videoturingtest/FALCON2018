#!/bin/bash
# FALCON: FAst and Lightweight CONvolution
#
# Authors:
#  - Chun Quan (quanchun@snu.ac.kr)
#  - U Kang (ukang@snu.ac.kr)
#  - Data Mining Lab. at Seoul National University.
#
# File: scripts/test.sh
#  - Test trained model saved in ../train_test/trained_model.
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

# dataset information
DATA="cifar100" #option: cifar10/cifar100

# model information
CONV="FALCON" # option: FALCON/MobileConv/StandardConv
MODEL="MobileNet" # option: MobileNet/VGG16

# argument
ARGS="-conv $CONV \
-m $MODEL \
-data $DATA \
"

python main.py $ARGS;
