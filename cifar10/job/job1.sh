#!/bin/sh



cd ..
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 1 --interval 20 --n-jobs 2 --num-gpus 2 \
--target cifar10_scd_v9_075.pkl

python get_adv.py --source cifar10_scd_v9_075.pkl --epsilon 1 --type scd

python train_scd_adv.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 1 --interval 20 --n-jobs 2 --num-gpus 2 \
--target cifar10_scd_v9_075_adv.pkl

python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2 \
--target cifar10_scd_v9_100.pkl

python train_scd_v11.py --nrows 0.1 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2 \
--target cifar10_scd_v11_adv.pkl --adv-train

python train_scd_v12.py --nrows 0.1 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
--updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2 \
--target cifar10_scd_v12_adv.pkl --adv-train

python train_scd_hl.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 \
--num-iters 1000 --hidden-nodes 20 --updated-features 128 --round 1 --interval 10 \
--n-jobs 2 --num-gpus 2 --target cifar10_scd_v15_075.pkl

python train_scd_hl.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 \
--num-iters 1000 --hidden-nodes 20 --updated-features 128 --round 32 --interval 10 \
--n-jobs 2 --num-gpus 2 --target cifar10_scd_v15_075_32.pkl