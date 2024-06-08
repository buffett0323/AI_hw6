#!/bin/bash

python main.py \
    --exp_name "${1}" \
    --model_name "${2}" \
    --train \
    --wandb_token "0aad23b14e9c1e0e2342caaefbcf3c240a8a3e5e" \
    --num_epochs 1 \
