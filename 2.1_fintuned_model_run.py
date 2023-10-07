import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

adapters_name = "quakumei/Mistral-OpenOrca-ft-45it"
model_name = "Open-Orca/Mistral-7B-OpenOrca"

device = "cuda" # the device to load the model 

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)

model = PeftModel.from_pretrained(model, adapters_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")


text = """<|im_start|> user
Generate a midjourney prompt for A robot on a first date<|im_end|>
<|im_start|> assistant
"""

encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
model_input = encoded
model.to(device)
generated_ids = model.generate(**model_input, max_new_tokens=200, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

