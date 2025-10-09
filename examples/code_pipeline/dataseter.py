import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time

from tqdm import tqdm
from glob import glob
from datasets import load_dataset

def clean_parquet_file(input_path, output_path=None):
    with open(input_path, 'rb') as f:
        data = f.read()

    # Find the LAST occurrence of b'PAR1'
    last_par1 = data.rfind(b'PAR1')
    if last_par1 == -1:
        raise ValueError("Missing 'PAR1' magic bytes — not a valid Parquet file")

    # Parquet files should end exactly 4 bytes after the last 'PAR1'
    expected_end = last_par1 + 4
    if expected_end < len(data):
        print(f"Trimming {len(data) - expected_end} extra bytes from file: {input_path}")
        data = data[:expected_end]  # Trim off the garbage

    # Write cleaned file
    output_path = output_path or input_path
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Cleaned file written to: {output_path}")


def cleaner(language):
    os.makedirs(os.path.join(FILTER_PATH, language), exist_ok=True)
    for file in glob(f"{os.path.join(DATA_PATH, language)}/*.parquet"):
        filename = os.path.basename(file)
        clean_parquet_file(file, os.path.join(FILTER_PATH, language, filename))

# Define a function to plot stats and save to file
def plot_stats(stats, save_path):
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    # List of columns for stats
    stat_columns = ['quantile', 'practice', 'educational', 'clarity', 'quality']
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through the columns and axes
    for i, stat in enumerate(stat_columns):
        ax = axes[i]
        # Extract the relevant data from the DataFrame
        for language, counts in stats[stat].items():
            counts = pd.Series(counts).sort_index()  # Ensure the counts are sorted
            counts.plot(kind='bar', ax=ax, label=language, alpha=0.7)
        
        # Set labels and title
        ax.set_title(f'{stat.capitalize()} Distribution')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Frequency')
        ax.legend(title='Languages')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot to the specified file path
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()  # Close the plot to avoid displaying it in some environments

def get_stats(language):
    stats = {'quantile': {}, 'practice': {}, 'educational': {}, 'clarity': {}, 'quality': {}}
    practice, educational, clarity, quantile, quality = [], [], [], [], []

    for file in glob(f"{os.path.join(DATA_PATH, language)}/*.parquet"):
        dataset = load_dataset('parquet', data_files=file)['train']
        quantile.extend(dataset['quantile'])
        practice.extend(dataset['practice'])
        educational.extend(dataset['educational'])
        clarity.extend(dataset['clarity'])
        quality.extend(dataset['quality'])

    stats['quantile'][language] = pd.Series(quantile).value_counts().sort_index()
    stats['practice'][language] = pd.Series(practice).value_counts().sort_index()
    stats['educational'][language] = pd.Series(educational).value_counts().sort_index()
    stats['clarity'][language] = pd.Series(clarity).value_counts().sort_index()
    stats['quality'][language] = pd.Series(quality).value_counts().sort_index()

    stats = pd.DataFrame(stats)
    stats.to_json(os.path.join("/iopsstor/scratch/cscs/rmachace/dataset/starcoderdata/stats", f"{language}.json"))
    plot_stats(stats, os.path.join("/iopsstor/scratch/cscs/rmachace/dataset/starcoderdata/stats", f"{language}.png"))

def create_quantiles(language):
    print("Processing:", language)
    all_quality = []
    for file in glob(f"{os.path.join(DATA_PATH, language)}/*.parquet"):
        print(file)
        dataset = load_dataset('parquet', data_files=file)['train']
        all_quality.extend(dataset['quality'])
    _, bin_edges = pd.qcut(all_quality + np.round(np.random.normal(loc=0, scale=1, size=len(all_quality))).astype(int), 5, labels=False, retbins=True, duplicates='drop')

    for file in glob(f"{os.path.join(DATA_PATH, language)}/*.parquet"):
        print({os.path.join(DATA_PATH, language)})
        dataset = load_dataset('parquet', data_files=file)['train']
        if 'quantile' in dataset.column_names:
            dataset = dataset.remove_columns(["quantile"])
        dataset = dataset.add_column(
            "quantile", 
            pd.cut(dataset["quality"], bins=bin_edges, labels=False, right=True, include_lowest=True)
        )
        dataset.to_parquet(file)

def create_threshold_dataset(language, threshold):
    print("Processing:", language)
    new_dir = os.path.join(FILTER_PATH, language, f"threshold_{threshold}")
    os.makedirs(new_dir, exist_ok=True)

    for i, file in enumerate(glob(f"{os.path.join(DATA_PATH, language)}/*.parquet")):
        dataset = load_dataset('parquet', data_files=file)['train']
        dataset = dataset.filter(lambda x: x['quantile'] >= threshold)
        dataset.to_parquet(os.path.join(new_dir, f'data_{i}.parquet'))

if __name__ == "__main__":
    DATA_PATH = "/iopsstor/scratch/cscs/rmachace/dataset/starcoderdata/new"
    FILTER_PATH = "/iopsstor/scratch/cscs/rmachace/dataset/starcoderdata/filtered"

    languages = [
        "ada",
        "agda",
        "alloy",
        "antlr",
        "applescript",
        "assembly",
        "augeas",
        "awk",
        "batchfile",
        "bluespec",
        "c",
        "c-sharp",
        "clojure",
        "cmake",
        "coffeescript",
        "common-lisp",
        "cpp",
        "css",
        "cuda",
        "dart",
        "dockerfile",
        "elixir",
        "elm",
        "emacs-lisp",
        "erlang",
        "f-sharp",
        "fortran",
        "glsl",
        "go",
        "groovy",
        "haskell",
        "html",
        "idris",
        "isabelle",
        "java",
        "java-server-pages",
        "javascript",
        "json",
        "julia",
        "kotlin",
        "lean",
        "literate-agda",
        "literate-coffeescript",
        "literate-haskell",
        "lua",
        "makefile",
        "maple",
        "markdown",
        "mathematica",
        "matlab",
        "ocaml",
        "pascal",
        "perl",
        "php",
        "powershell",
        "prolog",
        "protocol-buffer",
        "python",
        "r",
        "racket",
        "restructuredtext",
        "rmarkdown",
        "ruby",
        "rust",
        "sas",
        "scala",
        "scheme",
        "shell",
        "smalltalk",
        "solidity",
        "sparql",
        "sql",
        "stan",
        "standard-ml",
        "stata",
        "systemverilog",
        "tcl",
        "tcsh",
        "tex",
        "thrift",
        "typescript",
        "verilog",
        "vhdl",
        "visual-basic",
        "xslt",
        "yacc",
        "yaml",
        "zig"
    ]

    # Copy and clean the parquet
    # for language in languages:
    #     cleaner(language)

    thresholds = [0, 1, 2, 3, 4]
    for language in languages:
        try:
            create_quantiles(language)
            get_stats(language)
            for threshold in thresholds:
                create_threshold_dataset(language, threshold)
        except Exception as e:
            print(f"FAILED FOR: {language}", e)