#!/bin/bash

python main.py \
    --model_name "${1}" \
    --inference_base_model \
    --wandb_token "${2}"