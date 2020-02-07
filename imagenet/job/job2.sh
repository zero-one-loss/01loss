#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N svm

cd ..
python svm.py --c 0.01 --target imagenet_svm.pkl
python get_adv.py --source imagenet_svm.pkl --epsilon 1 --type svm
python svm_ad.py --c 0.01 --target imagenet_svm_adv.pkl
python svm_boot.py --c 0.01 --votes 100 --ratio 0.66 --target imagenet_svm_boot.pkl
python train_mlp.py --hidden-nodes 20 --lr 0.01 --iters 200 --target imagenet_mlp.pkl