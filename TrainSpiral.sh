#!/bin/bash

# bash TrainSpiral.sh

# tensorboard --logdir=runs &

for i in 0 50 20 0 80
do
    printf "\n\tM-DYR-H WITH NOISE LEVEL OF $i\n\n" 
    python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
	    --epochs 300 --M 50 100 --noise-level $i --reg-term 1 --dataset CIFAR10 --tb-dir "runs-${i}" --root-dir "datasets/spiral"
    printf "\n\tLLRM-DYR-H WITH NOISE LEVEL OF $i\n\n" 
    python3 train.py --Mixup 'Hidden' --BootBeta 'Hard' --experiment-name 'LLRM-DYR-H' \
	    --epochs 300 --M 50 100 150 250 --noise-level $i --reg-term 1 --dataset CIFAR10 --tb-dir "runs-${i}" --root-dir "datasets/cifar10"
done