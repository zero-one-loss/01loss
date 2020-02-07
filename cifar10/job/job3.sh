#!/bin/sh



cd ..
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_100 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v11_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v12_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.0625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075_32 --random-sign 1

python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_100 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v11_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v12_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.03125 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075_32 --random-sign 1

python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_075_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v9_100 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v11_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v12_adv --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075 --random-sign 1
python cifar10_original_01_01.py --epsilon 0.015625 --Lambda 0.1 --gpu 0 --epoch 20 --aug-epoch 20 --lr 0.0001 --train-size 200 --target cifar10_scd_v15_075_32 --random-sign 1