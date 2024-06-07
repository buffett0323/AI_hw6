import gc
import os
import json
import utils
import torch
import wandb
from tqdm.auto import tqdm
from trl import DPOTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, TextStreamer



def DPO_train(args, output_dir):
    wandb.login(key=args.wandb_token)
    wandb.init(project="hw6_rlhf",
               name=f"{args.exp_name}_{args.model_name.split('/')[1]}")

    torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device\n")

    # Load dataset
    # ================================DO NOT CHANGE!================================
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    dataset = dataset.rename_column('question', 'prompt')
    dataset = dataset.train_test_split(test_size=0.01)

    with open("./test_prompt.json", 'r') as f:
        test_data = json.load(f)
    # ================================DO NOT CHANGE!================================

    # Model
    # model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model_name,...)
    utils.YOUR_CODE_HERE

    # Perform model patching and add fast LoRA weights
    # model = FastLanguageModel.get_peft_model(model,...)
    utils.YOUR_CODE_HERE

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=output_dir,
        save_strategy=args.save_strategy,
        report_to=args.report_to
    )

    # Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=utils.YOUR_CODE_HERE,
        eval_dataset=utils.YOUR_CODE_HERE,
        args=training_args,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    # Save model
    model.save_pretrained(output_dir)

    # Inference
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)

    output_data = []

    for data in tqdm(test_data):
        print("=============Generated Answer After Fine-tuning=============\n")
        print(f"Question {data['id']}:\n"+data["prompt"])
        prompt = utils.alpaca_prompt.format(
            "You are a helpful assistant chatbot.",  # Instruction
            data["prompt"],  # Input
            "",  # Response, leave empty for generation
        )
        prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_sequences = model.generate(**prompt, streamer=text_streamer,
                           max_new_tokens=500)
        # Decode the generated output
        generated_text = tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True)[0]
        print("==============================================================\n")

        # Store the output in a list
        output_data.append({
            "id": data["id"],
            "prompt": data["prompt"],
            "generated_text": generated_text
        })
    
    # Ensure the submission directory exists
    submission_dir = "submission"
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    # Write the output data to a JSON file
    output_file = os.path.join(submission_dir, f"DPO_{args.model_name.split('/')[1]}.json")
    utils.write_json(output_data, output_file)

    # Flush memory
    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()
