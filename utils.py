import json

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_prompt(sample, EOS_TOKEN):
    instruction = sample["system"]
    input = sample["question"]
    accepted = sample["chosen"]
    rejected = sample["rejected"]
    sample["prompt"] = alpaca_prompt.format(instruction, input, "")
    sample["chosen"] = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample


def write_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)


YOUR_CODE_HERE = "*** YOUR CODE HERE ***"
