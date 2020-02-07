#!/bin/sh

#$-q datasci
#$-q datasci3
#$-cwd
#$-N scd_attack

cd ..
#python train_scd_hl.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 \
#--num-iters 1000 --hidden-nodes 20 --updated-features 128 --round 32 --interval 10 \
#--n-jobs 2 --num-gpus 2 --target imagenet_scd_v15_075_32.pkl

#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_100 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v11_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v12_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.0625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075_32 --random-sign 1
##
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_100 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v11_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v12_adv --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075 --random-sign 1
#python imagenet_original_01_01.py --epsilon 0.03125 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075_32 --random-sign 1
#
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075 --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_075_adv --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v9_100 --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v11_adv --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v12_adv --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075 --random-sign 1
python imagenet_original_01_01.py --epsilon 0.015625 --Lambda 0.01 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target imagenet_scd_v15_075_32 --random-sign 1