#!/bin/sh


cd ..

python cifar10_original_svm_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_adv --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_boot --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_mlp --random-sign 1

python cifar10_original_svm_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_adv --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_boot --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_mlp --random-sign 1

python cifar10_original_svm_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_adv --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_svm_boot --random-sign 1
python cifar10_original_svm_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_mlp --random-sign 1