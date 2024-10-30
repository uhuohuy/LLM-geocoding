import sys
import time
from pathlib import Path
from typing import Literal, Optional
import codecs
from genericpath import isfile
from os import listdir
import json
import string

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


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

def insert_multiple_strings(original_string, strings_to_insert, insertion_indices):
    result = []
    previous_index = 0
    
    for index, insert_string in zip(insertion_indices, strings_to_insert):
        result.append(original_string[previous_index:index])
        result.append(insert_string)
        previous_index = index
    
    result.append(original_string[previous_index:])
    
    return ''.join(result)


def remove_punctuation(input_string):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Use the translate method to remove punctuation
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string


def evalute(model, tokenizer, fabric, max_returned_tokens, temperature, top_k, prompt, input_string):
    sample = {"instruction": prompt, "input": input_string}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    return str(output)
    # fabric.print(output)


def main(
    INSTRUCT: str = "Identify the full address of the place name (marked with <START> <END>) in the text",
    input: str = """Cookville Volunteer Fire Department hosts 19th annual fundraiser Saturday. The Cookville Volunteer Fire Department will host its 19th annual fundraiser Saturday, March 28, from 12 p.m. to 8 p.m. at the fire station in Cookville, located on County Road 4045, behind the <START>Cookville<END> store. According to Cookville Fire Chief Wesley McCollum, patrons attending can expect a family-friendly environment with a hip hop for the kids and dominoes for the adults, as well as a scrumptious spread including brisket, crawfish, baked beans, potato salad and desserts prepared by volunteers, families and community members. A pie and cake auction will also be held at 6 p.m., McCollum said, and door prizes will be given away. "We have tons of door prizes people have donated, and we will probably be giving away around four an hour," McCollum said. The event will be free to the public but donations will be accepted and appreciated, he added. "We know not everyone can afford to donate right now, but we want to welcome everyone to come out and visit with us and enjoy some good food and fun,""",
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    max_char: int = 1000,
    precision: Optional[str] = None,
    marker: int = 102,
    max_seq_length: int = 512
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    # prompt_length = encoded.size(0)
    # max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_seq_length
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    merge_lora_weights(model)
    model = fabric.setup(model)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    
    
    output_file = str(checkpoint_dir) + str(lora_path)  + str(marker)
    output_file = remove_punctuation(output_file)
    print(output_file)

    '''process files of test data'''
    
    start_time = time.time()
    data_list = ['trnews','19th','wotr','geocorpora','gwn','LDC','wiktor','geovirus','TUD','neel','semeval']        
    data_list = ['trnews']
    for test_data in data_list: #'lgl','neel', 'wiktor'
        target_file = 'result/'+output_file+'_'+test_data+".json"
        total_return_results = {}
        io =  open('data/'+test_data+'.json',"r")
        true_dict = json.load(io)
        except_count = 0
        try:
            io = open(target_file,"r")
            total_return_results = json.load(io)
        except:
            total_return_results = {}            

        # Example usage of the geoparse function below reading from a directory and parsing all files.
        count = 0
        file_count = 0

        directory =  'data/'+ test_data+ "/" 
        files = [f for f in listdir(directory) if isfile(directory + f)]
        for f in files:
            count += 1
            #if f != '1030005.txt':
            #    continue
            
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
                text = insert_multiple_strings(total_line, ['<START>','<END>'],[int(place['start']), int(place['end'])])
                text = find_centered_substring(text, max_char)
                new_INSTRUCT = INSTRUCT.replace('the place name', place['LOC'] )
                
                address = evalute(model, tokenizer, fabric, max_seq_length, temperature, top_k, new_INSTRUCT,  text)
                # t = time.perf_counter() - t0
                # result = *evaluation_generator
                # address = str(*evaluation_generator).replace("</s>", "")
                print(place['LOC'], address)
                return_objects.append({'start':place['start'],'end':place['end'],'LOC':place['LOC'],'address':address})

                # try:
                # lat_str, lon_str = address.split(', ')
                #     latitude = float(lat_str)
                #     longitude = float(lon_str)
                # except ValueError:
                #     # Handle the exception (e.g., invalid format)
                #     latitude = 0
                #     longitude = 0
                # return_objects.append({'start':place['start'],'end':place['end'],'LOC':place['LOC'],'lat':latitude, 'lon': longitude})

                # c = 0
                # for value in evaluation_generator:
                #     print(c, value)
                #     c+=1
            if ID not in total_return_results:
                total_return_results[ID] = return_objects
            else:
                temp = total_return_results[ID]
                temp.extend(return_objects)
                total_return_results[ID] = temp
            f = open('result/'+output_file+'_'+test_data+".json", "w")
            json.dump(total_return_results, f)
            f.close()
        f = open('result/'+output_file+'_'+test_data+".json", "w")
        json.dump(total_return_results, f)
        f.close()
    end_time = time.time()
    print('total consumped time', end_time-start_time)
    #print(torch.cuda.max_memory_allocated)
    #print('memory usage', torch.cuda.max_memory_allocated() / 1e9:.02f)

    # tokens_generated = y.size(0) - prompt_length
    # fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
         fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
