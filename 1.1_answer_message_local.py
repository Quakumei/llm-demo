import openai
import sys

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Completion API
stream = False
prompt = sys.argv[1]
template = f"""
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

completion = openai.Completion.create(
    model=model,
    prompt=template,
    # messages=[{"role":"user", "content": f"{template}"}],
    echo=False,
    n=1, # choices count
    stream=stream,
    max_tokens=1024,
    logprobs=3)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)

print("\n\n" + "="*80)
print("Prompt:\n")
print("-----")
print(prompt)
print("-----")
print("\n" + "="*80)
print("Model completion\n")
print("-----")
from pprint import pprint
pprint(completion['choices'][0]['text'])
print("-----")

