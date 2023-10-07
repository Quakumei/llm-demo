# Chat finetuning demo

## Setup

Before running the scripts in the repo, please provide
huggingfaces Access Token for write to create repo for 
finetuned model. 

## 1. Running model out-of-box

There are three main models, which I can recommend to try out

Checkout https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#api-server to know
more about inference server

1. Mistral OpenOrca version (Best out-of-box)

```bash
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model Open-Orca/Mistral-7B-OpenOrca 
python 1.1_answer_message_local.py

# ... or run quantized example (don't forget to pip install autoawq)
python openorca-mistral-7b-awq.py
```

2. Basic version of Mistral

```bash
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model mistralai/Mistral-7B-v0.1
...
```

3. Instruct version of Mistral

```bash
python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.1
...
```

## 2. Finetune the model for downstreamtask

Example is QLoRa on MOpen-Orca/Mistral-7B-OpenOrca

```bash
huggingface-cli login # use your token to allow model saving
# Command to finetune the model, specify repo_id with your hugginface account name. 

# Get dataset of midjourney prompts...
# Be sure to match IO formats
git clone https://github.com/joshbickett/finetune-llama-2.git
mv finetune-llama-2/train.csv ./train.csv

# Looks for train.csv in --data_path to train on.
autotrain llm --train --project_name mistral-7b-mj-finetuned --model Open-Orca/Mistral-7B-OpenOrca --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 12 --num_train_epochs 45 --trainer sft --target_modules q_proj,v_proj --push_to_hub --repo_id quakumei/Mistral-OpenOrca-ft-45it
```

The result of the autotrain is an adapter model, which is used with base model to get predictions. 

```bash
python 2.1_fintuned_model_run.py  
```

After finetuning of the model, you run the model

