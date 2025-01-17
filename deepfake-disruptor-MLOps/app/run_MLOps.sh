#!/bin/bash

config="config/config.yaml"
step=$1


# step 0. data split trian, test
if [ $step -eq 0 ]; then
    taskset -c 1-16 python preprocessing.py \
	    -d $2 || exit 1
fi

# step 1. train
# If use the checkpoint model : option -p PATH
# EX) -p chkpt/chkpt_1000.pt
if [ $step -eq 1 ]; then
  taskset -c 1-16 python trainer.py \
    -i $config \
    -p $2 \
    -s $3  || exit 1
fi

## step 2. inference
#if [ $step -eq 2 ]; then
#
#fi
