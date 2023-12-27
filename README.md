# LLM-geocoding
This is the code for fine-tuning Mistral, llama2, Baichuan2, and Falcon for geocodng or toponym resolution task. These models are fine-tuned to accurately estimate toponyms' full addresses (e.g., city, state, country), subsequently determining their geo-coordinates via geocoders, as shown in the figure below.

<p align="center">
<a href="url">
 <img src="figure/llm-workflow.png"  ></a>
</p>

## Data Preparation
## Fine-tuning
### Mistral and Llama2
```shell
OUT='PLACE'
mkdir $OUT
python finetune_llama.py \
       --data_file "training_data7.json" \
       --optim 1 \
       --R 16 \
       --batch 32 \
       --ALPHA 16 \
       --dropout 0.1 \
       --BASE_MODEL "../llama-recipes/output/llama-2-7b-chat" \
       --OUTPUT_DIR=$OUT \
       --LEARNING_RATE 3e-3 \
       --neftune_noise_alpha 0.1 \
       --TRAIN_STEPS 800
```
