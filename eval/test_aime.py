import torch
import numpy as np
from src.services.nanovllm_v2 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset


temperature = 0.6

class Dataset_with_template(Dataset):
    def __init__(self, local_dir, data_source, tokenizer):
        self.dataframe = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(local_dir, data_source + ".parquet"),
            split="train",
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row_dict = self.dataframe[idx]
        prompt = row_dict["prompt"]
        raw_prompt = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        row_dict["raw_prompt"] = raw_prompt
        return row_dict

    def __len__(self):
        return len(self.dataframe)


def generate_answer(local_dir="datasets", model_path="/home/yyx/models/Qwen3-4B"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6 ,top_k=20, top_p=0.95, max_tokens=1024)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    dataset = Dataset_with_template(local_dir, "aime_2024", tokenizer)
    sample = dataset[5]
    prompt = sample["raw_prompt"]
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    outputs = llm.generate([prompt], sampling_params)

    output_ids = outputs[0]["token_ids"]
    print(f"total output tokens {len(output_ids)}")

    all_text = prompt + outputs[0]["text"]
    # generated_text = outputs[0]["text"]
    with open("aime_5_answer", "w") as f:
        f.write(all_text)
    

def main():
    generate_answer()

if __name__ == "__main__":
    main()
