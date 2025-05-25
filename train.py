from datasets import load_dataset
from rich import print
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# download dataset
dataset = load_dataset("neuralwork/fashion-style-instruct", split="train", token=HF_TOKEN)
print(dataset)

# print a sample triplet
print(dataset[0])

def format_instruction(sample):
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
        ### Input:
        {sample["input"]}

        ### Context:
        {sample["context"]}

        ### Response:
        {sample["completion"]}
    """

sample = dataset[0]
print(format_instruction(sample))

# BitsAndBytesConfig to quantize the model int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# base model id to fine-tune
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# load model 
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    use_cache=False, 
    device_map="auto"
)
model.config.pretraining_tp = 1

# load tokenizer, pad short samples with end of sentence token
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# get frozen vs trainable model param statistics
print_trainable_parameters(model)

# Supervised Fine-Tuning Trainer
training_args = SFTConfig(
    max_length=2048,
    output_dir="Llama-3.2-1B-Instruct-style",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=format_instruction,
)

# train
trainer.train()

# save model to output_dir in TrainingArguments
trainer.save_model()
