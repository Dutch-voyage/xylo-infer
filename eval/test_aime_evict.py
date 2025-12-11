import torch
import numpy as np
from src.services.nanovllm_v6 import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import datasets
from torch.utils.data import DataLoader, Dataset

from .utils import evaluate


selected_indices = [0, 1, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 23, 24, 26]

temperature = 0.6

class Dataset_with_template(Dataset):
    def __init__(self, local_dir, data_source, tokenizer):
        self.dataframe = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(local_dir, data_source + ".parquet"),
            split="train",
        )
        # pruned dataset for test_aime_evict.py
        self.dataframe = self.dataframe.select(selected_indices)        
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
    sampling_params = SamplingParams(temperature=0.6 ,top_k=20, top_p=0.95, max_tokens=16384)
    # temperature < 0 for greedy_sampling
    # sampling_params = SamplingParams(temperature=-1, max_tokens=1024)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    dataset = Dataset_with_template(local_dir, "aime_2024", tokenizer)
    
    batch_size = 18
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    scores = 0.0
    generate_lengths = 0
    length_list = []
    # logits_list = []
    token_id_list = []
    
    filename = "aime_rkv_1024_128"
    
    for batch in dataloader:
        prompts = batch["raw_prompt"]
        ground_truth = batch["ground_truth"]
        
        outputs = llm.generate(prompts, sampling_params)

        for idx in range(batch_size):
            output_ids = outputs[idx]["token_ids"]
            print(f"total output tokens {len(output_ids)}")

            length_list.append(len(output_ids))
            # logits_list.append(outputs[idx]["logits"])
            token_id_list.append(output_ids)

            score = evaluate(outputs[idx]["text"], ground_truth[idx])
            scores += score
            
            generate_length = len(output_ids)
            generate_lengths += generate_length

            all_text = prompts[idx] + outputs[idx]["text"] + "\n\n" + "score: " + str(score) + "\n\n" + "generated_tokens: " + str(generate_length) + "\n\n" + "=" * 100 + "\n\n"
            # generated_text = outputs[0]["text"]
            with open(filename, "a") as f:
                f.write(all_text)
    
    np.save(f"collected_data/{filename}.npy", {"length_list": length_list, "token_id_list": token_id_list})
    
    summary = "Evaluation completed." + "\n\n"
    summary += "scores sum: " + str(scores) + "\n\n"
    summary += f"Average score: {scores / batch_size}" + "\n\n"
    summary += "Total generated tokens: " + str(generate_lengths) + "\n\n"
    summary += f"Average generated tokens: {generate_lengths / batch_size}" + "\n\n"
    with open(filename, "a") as f:
        f.write(summary)
    
    print("Evaluation completed.")
    print("scores sum:", scores)
    print(f"Average score: {scores / batch_size}")
    print("Total generated tokens:", generate_lengths)
    print(f"Average generated tokens: {generate_lengths / batch_size}")
    
    

def main():
    generate_answer()

if __name__ == "__main__":
    main()
