import datasets as ds
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()

def msgs_dict(prompt, answer):
    return [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": answer}
    ]

def get_qa_lksy_ru_instruct_gpt4(dataset_sample):
    prompt = f"{dataset_sample['instruction']} {dataset_sample['input']}"
    answer = dataset_sample["output"]
    return prompt, answer

def get_template_from_sample(dataset_sample, tokenizer):
    prompt,answer = get_qa_lksy_ru_instruct_gpt4(dataset_sample)
    return tokenizer.apply_chat_template(msgs_dict(prompt, answer), tokenize=False)

def apply_template_for_df(df, tokenizer):
    df['text'] = df.progress_apply(lambda x: get_template_from_sample(x ,tokenizer), axis=1)
    return df

if __name__ == '__main__':
    dataset = ds.load_dataset("lksy/ru_instruct_gpt4", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
    
    print("="*80)
    print(dataset)
    print(tokenizer)

    #dataset_sample, formatted = get_template_from_sample(dataset[0], tokenizer)
    #print("="*80)
    #print(f"{dataset_sample=}\n{formatted=}")
    
    ds = dataset.to_pandas()
    ds = apply_template_for_df(ds, tokenizer)
    
    print("="*80)
    print(ds.head())

    ds['text'].to_csv("lksy_ru_instruct_gpt4.csv")

