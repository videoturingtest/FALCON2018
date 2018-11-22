#!/bin/bash
# FALCON: FAst and Lightweight CONvolution
#
# Authors:
#  - Chun Quan (quanchun@snu.ac.kr)
#  - U Kang (ukang@snu.ac.kr)
#  - Data Mining Lab. at Seoul National University.
#
# File: scripts/train_rank.sh
#  - Train the rank model.
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

# dataset information
DATA="cifar10" # option: cifar10/cifar100

# model information
CONV="RankFALCON" # option: RankFALCON/RankMobileConv
MODEL="VGG16" # option: MobileNet/VGG16

# training argument
EPOCH=200
OPTIMIZER='SGD' # option: SGD/Adagrad/Adam/RMSprop
BATCH_SIZE=128
LR=1e-2
RANK=2 # option: 1/2/3/4
ALPHA=0.5 # option: ALPHA is in (0,1]

# argument
ARGS="-it \
-conv $CONV \
-m $MODEL \
-data $DATA \
-ep $EPOCH \
-op $OPTIMIZER \
-bs $BATCH_SIZE \
-lr $LR \
-lr_dec \
-k $RANK \
-al $ALPHA
"

python main.py $ARGS;
