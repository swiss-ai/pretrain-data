import argparse
import numpy as np
import pandas as pd
import datasets
import pathlib
import os

from typing import Any, Dict, List
from packaging.version import Version
from vllm import LLM, SamplingParams

# Set logging
datasets.disable_progress_bars()

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--GPUS", type=int)
parser.add_argument("--GPU_id", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--limit", type=int)
parser.add_argument("--language", type=str)
args = parser.parse_args()

GPUS = args.GPUS
GPU_id = args.GPU_id
batch_size = args.batch_size
language = args.language
limit = args.limit

files = f"./datasets/{language}"
outputs = f"./quality/{language}"
model = "Qwen/Qwen2.5-Coder-32B-Instruct"
truncation_size = 32000
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=7)
instruction = """
You will be given a source code, evaluate its quality according to the following criteria, where each can be scored on the discrete scale 0-9 (0=Worst, 9=Best):
1. Clarity (<CLARITY>): Is the code readable, supported by comments and functions properly?  
- Examples: Best: Clear function and variable names and helpful comments. Worst: Poorly factored, unnecessarily complex logic.
2. Good Practices (<PRACTICE>): Does the code use recommended best practices?  
- Examples: Best: Use of classes, error handling, and modular functions. Worst: No exception handling, hardcoded values, long functions, copypasta, many special cases, dead code, god object.
3. Educational Value (<EDUCATIONAL>): Is it usable for teaching relevant programming concepts?  
- Examples: Best: Demonstrates modularity or consistency with an appropriate principal (e.g. OOP, functional, etc.) with explanations. Worst: Complex code with no explanation or context, insecure practices, unneeded generality (YAGNI). 

Also include the difficulty (<DIFFICULTY>) of the code (0-2) according to the following categories:
0. Begginer: Suitable for introductory level programmers with basic programming knowledge.
- Examples: Code contains introductory concepts like if-else statements and loops.
1. Intermediate: Suitable for programmers with some knowledge and experience in programming.
- Examples: Code contains classes, basic algorithms and data structures.
2. Advanced: Suitable for profficient programmers with deep knowledge in the field.
- Examples: Code contains complex algorithms, design patterns and concurrency.

Output Format: Return ONLY the values seperated by space, nothing else, where generated output has to look like: <CLARITY> <PRACTICE> <EDUCATIONAL> <DIFFICULTY>"""

llm = LLM(
    model=model,
    download_dir="./models",
    tensor_parallel_size=1,
)


# Create a class to do batch inference.
def batch_inference(batch) -> Dict[str, list]:
    prompts = [
        [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": content
                if len(content) < truncation_size
                else content[:truncation_size],
            },
        ]
        for content in batch["content"]
    ]
    outputs = llm.chat(prompts, sampling_params, use_tqdm=False)
    output_list = [output.outputs[0].text for output in outputs]
    batch["output"] = output_list
    return batch


# Iterate over files
for i, file in enumerate(sorted(pathlib.Path(files).glob("*.parquet"))):
    # Break loop based on limit
    if i == int(limit):
        break
    print(
        f"-------------------------------------FILE: {file}-------------------------------------"
    )

    # Load the Parquet file into a Hugging Face Dataset
    dataset = datasets.load_dataset("parquet", data_files=str(file))
    dataset_shard = dataset["train"].shard(index=GPU_id, num_shards=GPUS)
    output_shard = dataset_shard.map(
        batch_inference, batched=True, batch_size=batch_size
    )
    filename = os.path.basename(file)
    output_shard.to_parquet(f"{outputs}/shard_{GPU_id}_{filename}")
