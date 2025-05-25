from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
model_name_or_path = "Llama-3.2-1B-Instruct-style"
device = "cuda" # or "cuda" if you have a GPU
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# push model and tokenizer to HF hub under your username
model.push_to_hub("Llama-3.2-1B-Instruct-style")
tokenizer.push_to_hub("Llama-3.2-1B-Instruct-style")
