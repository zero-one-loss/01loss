#!/bin/sh



cd ..
python svm.py --c 0.01 --target cifar10_svm.pkl
python get_adv.py --source cifar10_svm.pkl --epsilon 1 --type svm
python svm_ad.py --c 0.01 --target cifar10_svm_adv.pkl
python svm_boot.py --c 0.01 --votes 100 --ratio 0.66 --target cifar10_svm_boot.pkl
python train_mlp.py --hidden-nodes 20 --lr 0.01 --iters 200 --target cifar10_mlp.pkl