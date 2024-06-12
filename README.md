## Title
Artificial Homework 6
2024/06/12

## Arthors
B09208038 地理四 劉正悅 

## 💾 Installation Instructions
### Conda Installation
For the installation, I refer to the TA's instruction.
e.g. I select `pytorch-cuda=12.1` for CUDA 12.1. Replace with 11.8 or else if you have different version.

```bash
conda create --name unsloth_env python=3.10
conda activate unsloth_env

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb
```

### Problem solving
Please use `python -m xformers.info` to see whether the triton_available is True. If not, please do the following command.

```bash
conda uninstall triton -y
pip uninstall triton -y
conda uninstall torchtriton
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
```

### Requirements
Or you can try installing with the following command.
```bash 
pip install -r requirements.txt 
```

## Code running
I have added some parameters in the `run.sh` file, and i trained on GPU RTX 3090 machine. 
You can remove the batch size settings if it doesn't fit your machine.

### Training 
Replace the wandb_token with your token please !!!!
```bash
# Actually, the structure is bash run.sh "exp_name" "model_name" "wandb_token"
bash run.sh ORPO unsloth/mistral-7b-instruct-v0.3-bnb-4bit YOUR_WANDB_TOKEN
```

### Inference
```bash
# Actually, the structure is bash inference.sh "model_name" "wandb_token"
bash inference.sh unsloth/mistral-7b-v0.3-bnb-4bit YOUR_WANDB_TOKEN
```
