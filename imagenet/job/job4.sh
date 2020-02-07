#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N svm_attack

cd ..

#python imagenet_original_svm_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm_adv --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm_boot --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_mlp --random-sign 1


#python imagenet_original_svm_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm_adv --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_svm_boot --random-sign 1
#python imagenet_original_svm_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target imagenet_mlp --random-sign 1
#
#
python imagenet_original_svm_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_svm --random-sign 1
python imagenet_original_svm_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_svm_adv --random-sign 1
python imagenet_original_svm_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_svm_boot --random-sign 1
python imagenet_original_svm_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 1 --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_mlp --random-sign 1