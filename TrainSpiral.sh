#!/bin/bash

# bash TrainSpiral.sh

# tensorboard --logdir=runs &

for i in 0
do
    printf "\n\tM-DYR-H WITH NOISE LEVEL OF $i\n\n" 
    python3 train.py --Mixup 'Static' --experiment-name 'M-DYR-H' \
	    --epochs 100 --M 25 75 --noise-level $i --reg-term 1 --dataset "spiral" --tb-dir "runs-${i}" --root-dir "datasets/spiral"
done