#!/bin/bash

# bash TrainSpiral.sh

# tensorboard --logdir=runs &

for i in 0 20 50 80
do
    printf "\nNOISE LEVEL OF $i\n\n" 

    python3 train.py --Mixup 'Static' --experiment-name 'M-DYR-H' \
	     --epochs 100 --M 25 75 --noise-level $i --reg-term 1 --dataset "spiral" --datasets-dir "datasets"
    
    python3 train.py --Mixup 'Hidden' --experiment-name 'LRM-DYR-H' \
	    --epochs 100 --M 25 75 --noise-level $i --reg-term 1 --dataset "spiral" --datasets-dir "datasets"
done