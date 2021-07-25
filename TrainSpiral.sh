#!/bin/bash

# bash TrainSpiral.sh

# tensorboard --logdir=tensorboard &

EPOCHS=300
REG_TERM=1
DATASET="spiral"
DATASETS_DIR="datasets"


for NOISE_LEVEL in 0 20 50 80
do
    printf "\nNOISE LEVEL OF $NOISE_LEVEL\n\n" 

    python3 train.py --Mixup 'None' --experiment-name 'DYR-H' \
	     --epochs $EPOCHS --M 100 250 --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr 0.01 --dataset $DATASET --datasets-dir $DATASETS_DIR
    
    python3 train.py --Mixup 'True' --layer-mix-bounds 0 1 --experiment-name 'M-DYR-H' \
	     --epochs $EPOCHS --M 25 75 --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr 0.001 --dataset $DATASET --datasets-dir $DATASETS_DIR
    
    python3 train.py --Mixup 'True' --layer-mix-bounds 0 4 --experiment-name 'LRM-DYR-H' \
	     --epochs $EPOCHS --M 25 50 75  --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr 0.001 --dataset $DATASET --datasets-dir $DATASETS_DIR
    
    python3 train.py --Mixup 'True' --layer-mix-bounds 1 5 --experiment-name 'LLRM-DYR-H' \
	     --epochs $EPOCHS --M 25 50 75  --noise-level $NOISE_LEVEL --reg-term $REG_TERM --lr 0.001 --dataset $DATASET --datasets-dir $DATASETS_DIR
done