#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import codecs
from genericpath import isfile
from os import listdir
import json
import time
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except: 
    pass


import string

def find_centered_substring(text, max_chars=1000):
    # Find the position of <START> and <END>
    start_idx = text.find('<START>')
    end_idx = text.find('<END>')

    if start_idx != -1 and end_idx != -1:
        # Calculate the start and end character indices for the centered substring
        start_char_idx = max(start_idx - max_chars, 0)
        end_char_idx = min(end_idx + max_chars, len(text))

        # Find the nearest whitespace to the left of the start_char_idx
        while start_char_idx > 0 and text[start_char_idx] != ' ':
            start_char_idx -= 1

        # Find the nearest whitespace to the right of the end_char_idx
        while end_char_idx < len(text) - 1 and text[end_char_idx] != ' ':
            end_char_idx += 1

        # Extract the centered substring based on character indices
        centered_substring = text[start_char_idx:end_char_idx + 1]

        return centered_substring

    return ""
    
def clean_location_text(text: str) -> str:
    """
    Extract a clean geographic name from noisy text.
    Handles patterns like <START>, <END>, /******/, <s>, #, etc.
    """
    # 1️⃣ Split by newline and take the first non-empty part
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    if not parts:
        return ""
    first_part = parts[0]

    # 2️⃣ Remove <START> and <END> tags
    cleaned = re.sub(r"<START>|<END>", "", first_part)

    # 3️⃣ Remove common filler tokens: /******/, <s>, ###, etc.
    #    ⚠️ Don't remove letters like 's'!
    cleaned = re.sub(r"/\*+/", " ", cleaned)      # remove /***/
    cleaned = re.sub(r"<s>", " ", cleaned)        # remove <s>
    cleaned = re.sub(r"[#*]+", " ", cleaned)      # remove #### or ****

    # 4️⃣ Clean up extra spaces and punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-_").strip()

    return cleaned

def remove_consecutive_duplicates(text: str) -> str:
    """
    Remove only consecutive duplicate items from a comma-separated string.
    Example:
        'japan, japan, japan, asia' -> 'japan, asia'
        'paris, france, paris' -> 'paris, france, paris'
    """
    # Split and clean items
    items = [x.strip() for x in text.split(',') if x.strip()]
    
    if not items:
        return ""

    # Keep first item, skip consecutive duplicates (case-insensitive)
    result = [items[0]]
    for item in items[1:]:
        if item.lower() != result[-1].lower():
            result.append(item)
    
    return ', '.join(result)

def insert_tags(text, word, start_index, end_index, tolerance=15):
    """
    Inserts '<START>' before and '<END>' after the specified word in the text, 
    considering possible offsets in the provided start_index and end_index.
    
    Parameters:
    - text (str): The input string.
    - word (str): The word to surround with tags.
    - start_index (int): The provided starting index of the word.
    - end_index (int): The provided ending index of the word.
    - tolerance (int): The maximum offset allowed for start_index and end_index.
    
    Returns:
    - str: The modified string with tags inserted.
    """
    
    # Define the possible range of indices to search for the word
    search_start = max(0, start_index - tolerance)
    search_end = min(len(text), end_index + tolerance)
    
    # Search for the word within the specified range
    index = text.find(word, search_start, search_end)
    
    # If the word is found within the range, insert the tags
    if index != -1:
        start_index = index
        end_index = index + len(word)
        tagged_text = text[:start_index] + '<START>' + word + '<END>' + text[end_index:]
        return tagged_text
    else:
        # If the word is not found within the range, return the original text
        return text

def remove_punctuation(input_string):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Use the translate method to remove punctuation
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    output_file:  str = "",
    output_coor: bool = False,
    max_char: int = 1000,
    INSTRUCT = "Identify the full address of the place name (marked with <START> <END>) in the text",
    INPUT = "RT @akwasisarpong: An update on the health status of the US man on admission for  fever  at Nyaho Clinic will be helpful. #Ghana #ebola wat\u0085",
):
    print(base_model, lora_weights, max_char,load_8bit)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
        
    model = model.merge_and_unload()
    if device == 'cuda':
        model = model.cuda()


    # unwind broken decapoda-research config
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    model.config.pad_token_id = tokenizer.pad_token_id = pad_token_id  # unk
    model.config.bos_token_id = bos_token_id
    model.config.eos_token_id = eos_token_id

    if not load_8bit:
        model.half()  

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=30,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)

        yield prompter.get_response(output)
    place_count = 0
    count = 0    
    output_file = base_model + lora_weights  + str(output_file)
    output_file = remove_punctuation(output_file)
    print(output_file)
    start_time = time.time()
    '''process files of test data'''
    
    data_list = ['19th','wotr','trnews','geocorpora','gwn','LDC','wiktor']       # ,'geovirus','TUD','neel','semeval'
    for test_data in data_list: 
        target_file = 'data/'+output_file+'_'+test_data+".json"
        total_return_results = {}
        io =  open('data/'+test_data+'.json',"r")
        true_dict = json.load(io)
        except_count = 0
        try:
            io = open(target_file,"r")
            total_return_results = json.load(io)
        except:
            total_return_results = {}            

        count = 0
        file_count = 0

        directory =  'data/'+ test_data+ "/" 
        files = [f for f in listdir(directory) if isfile(directory + f)]
        for f in files:
            count += 1            
            total_line = ''
            ID = f[0:len(f)-4]
            if ID not in true_dict:
                continue
            if not true_dict[ID]:
                continue
            exist_places = []
            print(test_data, count,'#'*50, f)
            for line in codecs.open(directory + f, encoding="utf-8"):
                total_line += line
            # print(total_line)
            return_objects = []
            for place in true_dict[ID]:
                bool_exist = 0
                if ID in total_return_results:
                    for exist_place in total_return_results[ID]:
                        if int(exist_place['start']) == int(place['start']):
                            bool_exist = 1
                            break
                if bool_exist:
                    continue
                text = insert_tags(total_line,place['LOC'],int(place['start']), int(place['end']))
                text = find_centered_substring(text, max_char)
                new_INSTRUCT = INSTRUCT.replace('the place name', place['LOC'])
                evaluation_generator = evaluate(new_INSTRUCT, text)
                place_count +=1
                address = str(*evaluation_generator).replace("</s>", "")
                address = clean_location_text(address)
                address = remove_consecutive_duplicates(address)
                if not output_coor:
                    return_objects.append({'start':place['start'],'end':place['end'],'LOC':place['LOC'],'address':address})
                else:
                    try:
                        lat_str, lon_str = address.split(', ')
                        latitude = float(lat_str)
                        longitude = float(lon_str)
                    except ValueError:
                        # Handle the exception (e.g., invalid format)
                        latitude = 0
                        longitude = 0
                    return_objects.append({'start':place['start'],'end':place['end'],'LOC':place['LOC'],'lat':latitude, 'lon': longitude})

            if ID not in total_return_results:
                total_return_results[ID] = return_objects
            else:
                temp = total_return_results[ID]
                temp.extend(return_objects)
                total_return_results[ID] = temp
            f = open(target_file, "w")
            json.dump(total_return_results, f)
            f.close()
        f = open(target_file, "w")
        json.dump(total_return_results, f)
        f.close()
    end_time = time.time()
    print('total consumped time', end_time-start_time)
    print('total places:', place_count)
if __name__ == "__main__":
    fire.Fire(main)
