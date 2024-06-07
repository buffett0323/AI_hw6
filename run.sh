#!/bin/bash

python main.py \
    --exp_name "${1}" \
    --model_name "${2}" \
    --train \
    --wandb_token "${3}" \
    --num_epochs 1 \
