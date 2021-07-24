#!/bin/bash

# bash TrainSpiral.sh

# tensorboard --logdir=tensorboard &

LR=0.001
EPOCHS=100
REG_TERM=1

DATASET="spiral"
DATASETS_DIR="datasets"


for NOISE_LEVEL in 20 50 80
do
    printf "\nNOISE LEVEL OF $NOISE_LEVEL\n\n" 

    python3 train.py --Mixup 'None' --experiment-name 'DYR-H' \
	     --epochs $EPOCHS --M 50 75 --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr $LR --dataset $DATASET --datasets-dir $DATASETS_DIR
    
    python3 train.py --Mixup 'Static' --experiment-name 'M-DYR-H' \
	     --epochs $EPOCHS --M 25 75 --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr $LR --dataset $DATASET --datasets-dir $DATASETS_DIR
    
    python3 train.py --Mixup 'Hidden' --experiment-name 'LRM-DYR-H' \
	     --epochs $EPOCHS --M 25 50 75  --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr $LR --dataset $DATASET --datasets-dir $DATASETS_DIR
done