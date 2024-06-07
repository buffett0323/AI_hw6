import gc
import os
import json
import utils
import torch
import wandb
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth import is_bfloat16_supported


def LLM_inference(args):
    wandb.login(key=args.wandb_token)
    wandb.init(project="hw6_rlhf",
               name=f"{args.exp_name}_{args.model_name.split('/')[1]}")
    torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device\n")

    # Load dataset
    # ================================DO NOT CHANGE!================================
    with open("./test_prompt.json", 'r') as f:
        test_data = json.load(f)
    # ================================DO NOT CHANGE!================================

    # Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=torch_dtype,
        load_in_4bit=True,
    )

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
    output_file = os.path.join(submission_dir, f"{args.model_name.split('/')[1]}.json")
    utils.write_json(output_data, output_file)

    # Flush memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
