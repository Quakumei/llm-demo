from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

quant_path = "TheBloke/Mistral-7B-OpenOrca-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
torch.cuda.empty_cache()
gc.collect()

print("loaded model...")

# Convert prompt to tokens
prompt_template = """\
<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""

tokens = tokenizer(
        prompt_template.format(prompt="Continue the phrase: never gonna give you up, "), 
    return_tensors='pt'
).input_ids.cuda()
gc.collect()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)
