#!/bin/bash
# FALCON: FAst and Lightweight CONvolution
#
# Authors:
#  - Chun Quan (quanchun@snu.ac.kr)
#  - U Kang (ukang@snu.ac.kr)
#  - Data Mining Lab. at Seoul National University.
#
# File: scripts/train.sh
#  - Train the model.
#
# Version: 1.0s
#==========================================================================================
cd ../src/train_test

# dataset information
DATA="cifar10" #option: cifar10/cifar100

# model information
CONV="FALCON" # option: FALCON/MobileConv/StandardConv
MODEL="VGG16" # option: MobileNet/VGG16

# training argument
EPOCH=200
OPTIMIZER='SGD' # option: SGD/Adagrad/Adam/RMSprop
BATCH_SIZE=128
LR=1e-2

# argument
ARGS="-it \
-conv $CONV \
-m $MODEL \
-data $DATA \
-ep $EPOCH \
-op $OPTIMIZER \
-bs $BATCH_SIZE \
-lr $LR \
-lr_dec
"

python main.py $ARGS;
