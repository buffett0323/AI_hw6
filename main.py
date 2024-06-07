import DPO
import ORPO
import time
import logging
import inference
import argparse
from pathlib import Path


def log_hyperparameters(args):
    logging.info("Hyperparameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        choices=["DPO", "ORPO"])
    parser.add_argument("--model_name", type=str,
                        choices=["unsloth/llama-3-8b-bnb-4bit",
                                 "unsloth/mistral-7b-v0.3-bnb-4bit"],
                        required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference_base_model", action="store_true")
    parser.add_argument("--wandb_token", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str,
                        default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--max_steps", type=int, default=0, choices=[500, 1000, 1500])
    parser.add_argument("--num_epochs", type=int, choices=[1, 3, 5])
    parser.add_argument("--optimizer", type=str, default="paged_adamw_32bit",
                        choices=["paged_adamw_32bit", "paged_adamw_8bit"])
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--logging_strategy", type=str,
                        default="steps", choices=["steps", "epoch"])
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--evaluation_strategy", type=str,
                        default="steps", choices=["steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--report_to", type=str, default="wandb")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create a timestamp
    current_time = time.strftime("%Y%m%d-%H%M%S")
    print(f"Current time: {current_time}\n")

    # Create the output directory path
    output_dir = Path(f"{args.output_dir}/{args.exp_name}_{current_time}")
    
    # Create the directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory at: {output_dir}\n")

    # Set up logging
    log_file_name = output_dir / f"{args.exp_name}-{current_time}.log"
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    
    log_hyperparameters(args)

    if args.train:
        if args.exp_name == "DPO":
            DPO.DPO_train(args, output_dir)
        elif args.exp_name == "ORPO":
            ORPO.ORPO_train(args, output_dir)
        else:
            raise ValueError("Invalid experiment name")
    
    if args.inference_base_model:
        if args.model_name == "unsloth/llama-3-8b-bnb-4bit":
            print("Inference with base model: unsloth/llama-3-8b-bnb-4bit")
            inference.LLM_inference(args)
        elif args.model_name == "unsloth/mistral-7b-v0.3-bnb-4bit":
            print("Inference with base model: unsloth/mistral-7b-v0.3-bnb-4bit")
            inference.LLM_inference(args)
        else:
            raise ValueError("Invalid model name")
