import glob
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import glob
import os
import pandas as pd
import tqdm

from tqdm import tqdm
from collections import Counter
from datasets import load_dataset


def sample_languages(languages, data_root, threshold, amount):
    for language in languages:
        for file in glob.glob(f"{data_root}/{language}/threshold_{threshold}/*"):
            orig_dataset = load_dataset('parquet', data_files=file)["train"]
            orig_dataset = orig_dataset.filter(lambda example: example['quantile'] == threshold)
            # Print the first 5 samples
            for i, content in enumerate(orig_dataset['content']):
                print("--------------------------------------------------------------")
                print(content)
                if i == amount:
                    break    

def fasttext_create_dataset(data_path, text_column, label_column):
    """
    Creates a fasttext dataset from the given Parquet files and splits it into train/test sets.
    """
    def format_to_fasttext(text, label):
        return f"__label__{int(label)} {repr(text)}\n"

    def write_to_file(file_path, data):
        with open(file_path, "w") as file:
            for line in data:
                file.write(line)

    fasttext_dir = f"{os.path.dirname(data_path)}/fasttext/{label_column}"
    os.makedirs(fasttext_dir, exist_ok=True)
    all_data = []
    
    # Loop through all parquet files in the data_path directory
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(fasttext_dir, file_name + ".txt")
    
    # Read the Parquet file
    dataset = pd.read_parquet(data_path)
    dataset['content'] = dataset['content'].str.replace('__label__', 'LABEL', regex=False)
    fasttext_data = [
        format_to_fasttext(
            example[text_column], example[label_column]
        ) for idx, example in dataset.iterrows()
    ]
    write_to_file(output_file, fasttext_data)
    return fasttext_dir, output_file


def fasttext_create_dataset_multi(data_path, text_column, label_columns):
    """
    Creates a fasttext dataset from the given Parquet files and splits it into train/test sets.
    """
    def format_to_fasttext(text, labels):
        res = ""
        for label in labels:
            res = res + f"__label__{int(label)} "
        return res + f"{repr(text)}\n" #[1:-1]?

    def write_to_file(file_path, data):
        with open(file_path, "w") as file:
            for line in data:
                file.write(line)

    fasttext_dir = f"{os.path.dirname(data_path)}/fasttext/"
    os.makedirs(fasttext_dir, exist_ok=True)
    all_data = []
    
    # Loop through all parquet files in the data_path directory
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(fasttext_dir, file_name + ".txt")
    
    # Read the Parquet file
    dataset = pd.read_parquet(data_path)
    # REPLACE
    dataset['content'] = dataset['content'].str.replace('__label__', 'LABEL', regex=False)
    fasttext_data = [format_to_fasttext(example[text_column], example[label_columns].values.tolist()) for idx, example in dataset.iterrows()]
    write_to_file(output_file, fasttext_data)
    print(f"Written to: {output_file}")
    return fasttext_dir, output_file


def fasttext_split_dataset(input_file, train_ratio):
    # Read
    input_folder = os.path.dirname(input_file)
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Split
    split_index = int(len(lines) * train_ratio)
    train_data = lines[:split_index]
    test_data = lines[split_index:]

    # Write
    with open(os.path.join(input_folder, 'train.txt'), 'w') as f:
        f.writelines(train_data)
    with open(os.path.join(input_folder, 'test.txt'), 'w') as f:
        f.writelines(test_data)
    return os.path.join(input_folder, 'train.txt'), os.path.join(input_folder, 'test.txt') 