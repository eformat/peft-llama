from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from random import randrange
import torch
import os

# fine-tuned model id
model_id = "eformat/Llama-3.2-1B-Instruct-style"

# load base LLM model, LoRA params and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def format_instruction(sample):
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        {sample["input"]}

        ### Context:
        {sample["context"]}

        ### Response:

    """

HF_TOKEN = os.getenv("HF_TOKEN")

# download dataset
dataset = load_dataset("neuralwork/fashion-style-instruct", split="train", token=HF_TOKEN)
print(dataset)

# select random sample
sample = dataset[randrange(len(dataset))]

# create prompt for inference
prompt = format_instruction(sample)
print(prompt)

device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenize input text
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
attention_mask = inputs["attention_mask"]


# inference, 5 outfit combinations make up around 700-750 tokens
with torch.inference_mode():
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=800, 
        do_sample=True, 
        top_p=0.9,
        temperature=0.9
    )

# decode token ids to text
outputs = outputs.detach().cpu().numpy()
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# outputs is a list of length num_prompts
# parse the completed part
output = outputs[0][len(prompt):]

print(f"Instruction: \n{sample['input']}\n")
print(f"Context: \n{sample['context']}\n")
print(f"Ground truth: \n{sample['completion']}\n")
print(f"Generated output: \n{output}\n\n\n")
