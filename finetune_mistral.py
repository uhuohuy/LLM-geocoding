#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import transformers
import textwrap
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import argparse

import fire
import torch
from datasets import load_dataset
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
import time
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)
start_time = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data_file', type=str, default="alpaca-chennai.json")
parser.add_argument('--BASE_MODEL', type=str, default="kittn/mistral-7B-v0.1-hf")
parser.add_argument('--OUTPUT_DIR', type=str, default="PLACE6")
parser.add_argument('--LEARNING_RATE', type=float, default=3e-4)
parser.add_argument('--TRAIN_STEPS', type=int, default=300)
parser.add_argument('--R', type=int, default=8)
parser.add_argument('--optim', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--ALPHA', type=int, default=16)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--neftune_noise_alpha', type=float, default=5)



CUTOFF_LEN = 1024

optims = {0:'adamw_hf', 1:'adamw_torch', 2:'adamw_torch_fused', 3:'adamw_apex_fused', 4:'adamw_anyprecision', 5:'adafactor'}

args = parser.parse_args()

print(args.data_file, args.neftune_noise_alpha, args.BASE_MODEL, args.batch, args.R, args.OUTPUT_DIR, args.LEARNING_RATE, args.dropout, args.ALPHA, args.TRAIN_STEPS, optims[args.optim])


model = LlamaForCausalLM.from_pretrained(
    args.BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    args.BASE_MODEL,
    model_max_length=CUTOFF_LEN,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token


# tokenizer = LlamaTokenizer.from_pretrained(args.BASE_MODEL)
 
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

data = load_dataset("json", data_files=args.data_file)

def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
 

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
 
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

totla_size = len(data['train'])
test_size = int(totla_size/10)
print(test_size)
train_val = data["train"].train_test_split(
    test_size=test_size, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

print(train_data)

LORA_R = args.R
LORA_ALPHA = args.ALPHA
LORA_DROPOUT= args.dropout
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = args.batch
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
TRAIN_STEPS = args.TRAIN_STEPS


model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    #warmup_ratio= 0.3,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim=optims[args.optim],
    evaluation_strategy="steps",
    save_strategy="steps",
    neftune_noise_alpha=args.neftune_noise_alpha,
    eval_steps=2,
    save_steps=2,
    output_dir=args.OUTPUT_DIR,
    save_total_limit=360,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict

 
model = torch.compile(model)
 
trainer.train()
model.save_pretrained(args.OUTPUT_DIR)
end_time = time.time()
print('totol_time', end_time-start_time, 'seconds')

