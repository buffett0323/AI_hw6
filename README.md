# Artificial Homework 6

**Date**: 2024/06/12

**Author**: B09208038 地理四 劉正悅

---

## 💾 Installation Instructions

### Conda Installation

For the installation, I refer to the TA's instruction. For example, I select `pytorch-cuda=12.1` for CUDA 12.1. Replace with 11.8 or another version if you have a different version.

```bash
conda create --name unsloth_env python=3.10
conda activate unsloth_env

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb
```

### Problem Solving

Please use `python -m xformers.info` to check if `triton_available` is `True`. If not, please run the following commands:

```bash
conda uninstall triton -y
pip uninstall triton -y
conda uninstall torchtriton
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
```

### Requirements

Alternatively, you can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Code Running

I have added some parameters in the `run.sh` file, and I trained on a GPU RTX 3090 machine. You can remove the batch size settings if they don't fit your machine.

Please replace the `wandb_token` with your token.

### Training
```bash
# Structure: bash run.sh "exp_name" "model_name" "wandb_token"
bash run.sh ORPO unsloth/mistral-7b-instruct-v0.3-bnb-4bit YOUR_WANDB_TOKEN
```
---

### Inference
```bash
# Structure: bash inference.sh "model_name" "wandb_token"
bash inference.sh unsloth/mistral-7b-v0.3-bnb-4bit YOUR_WANDB_TOKEN
```
---