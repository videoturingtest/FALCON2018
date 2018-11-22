FALCON: FAst and Lightweight CONvolution
===

This repositary provides implementations of FALCON convolution / Mobile convolution and their corresponding CNN model.
FALCON, a faster and lighter convolution, is capable of compressing and accelerateing standard convolution. 
FALCON compress and accelerate MobileNet with standard convolution for 7.4× and 2.36×, respectively.

## Overview
#### Code structure
* `./src`: source code for FALCON
    * `./src/model`: python scripts for model definition

    * `./src/main`: python scripts for training/testing models defined in `./src/model`

    * `./src/utils`: utils for execution of training/testing codes in `./src/main`

* `./scripts`: shell scripts for execution of training/testing codes in `./main`

#### Naming convention
**FALCON**: FAst and Lightweight CONvolution - the new convolution architecture we proposed

**MobileConv**: Convolution architecture from paper 'MobileNet' (refer to https://arxiv.org/abs/1704.04861)

**Rank**: Rank of convolution. Copy the conv layer for n times, run independently and add output together at the end of the layer. This hyper-parameter helps balace compression rate/ accelerate rate and accuracy.

#### Data description
* CIFAR-10 datasets
* CIFAR-100 datasets
* Note that: The datasets depends on torchvision (https://pytorch.org/docs/stable/torchvision/datasets.html#cifar). You don't have to download anything. When execute the source code, the datasets will be automaticly download if it is not detected.

#### Output
* After training, the trained model will be saved in `src/train_test/trained_model/`.
* `You can test the model only if there is a trained model in src/train_test/trained_model/.`

## Install
#### Environment 
* Unbuntu
* CUDA 9.0
* Python 3.6
* torch
* torchvision
#### Dependence Install
    pip3 install torch torchvision

## How to use 
#### Clone the repository
    git clone https://github.com/quanchun/FALCON
    cd FALCON
#### DEMO
* To train the model, run script:
    ```    
    cd scr/train_test
    python main.py -train -conv StandardConv
    python main.py -train -conv FALCON
    ```
    The trained model will be saved in src/train_test/trained_model/
* To test the model, run script:
    ```
    cd scr/train_test
    python main.py -conv StandardConv
    python main.py -conv FALCON
    ```
    The testing accuracy and inference time will be printed on the screen.
    `You can test the model only if there is a trained model in train_test/trained_model/.`
* To check the trained model size, run script:
    ```
    cd scr/train_test/trained_model
    ls -l
    ```
* Pre-trained model is saved in FALCON/src/train_test/trained_model/
    * Standard model: (It is about 115M. You have to train it first, since trained model is too large to upload.)
        `conv=StandardConv,model=MobileNet,data=cifar100,rank=1,alpha=1.pkl`
    * FALCON model: (It is trained and saved in folder.)
        `conv=FALCON,model=MobileNet,data=cifar100,rank=1,alpha=1.pkl`
####  Scripts
* There are four demo scripts: `scripts/train.sh`, `scripts/train_rank.sh`, `scripts/test.sh`, `scripts/test_rank.sh`
* You can change arguments in `.sh` files to train/test different model.
    * `train.sh`: Execute training of model (-conv FALCON -m VGG16 -data cifar10)
        * Output: trained model will be saved
        * Training procedure and result will be print on the screen.
    * `trian_rank.sh`: Execute training of model (-conv RankFALCON -m VGG16 -data cifar10 -k 2 -al 0.5)
        * Output: trained model will be saved
        * Training procedure and result will be print on the screen.
    * `test.sh`: Execute test of trained model (-conv FALCON -m MobileNet -data cifar100)
        * Accuracy/ inference time/ compression rate/ computation reduction rate will be print on the screen.
    * `test_rank.sh`: Execute test of trained model (-conv FALCON -m MobileNet -data cifar100 -k 2 -al 0.9)
        * Accuracy/ inference time/ compression rate/ computation reduction rate will be print on the screen.

## Contact us
- Chun Quan (quanchun@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.
