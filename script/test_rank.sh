#!/bin/bash
# FALCON: FAst and Lightweight CONvolution
#
# Authors:
#  - Chun Quan (quanchun@snu.ac.kr)
#  - U Kang (ukang@snu.ac.kr)
#  - Data Mining Lab. at Seoul National University.
#
# File: scripts/test_rank.sh
#  - Test trained rank model saved in ../src/train_test/trained_model.
#
# Version: 1.0
#==========================================================================================
cd ../src/train_test

# dataset information
DATA="cifar100" #option: cifar10/cifar100

# model information
CONV="RankFALCON" # option: RankFALCON/RankMobileConv
MODEL="MobileNet" # option: MobileNet/VGG16

# train argument
ALPHA=0.9 # option: ALPHA is in (0,1]
RANK=2 # option: 1/2/3/4

# argument
ARGS="-conv $CONV \
-m $MODEL \
-data $DATA \
-al $ALPHA
-k $RANK
"

python main.py $ARGS;
