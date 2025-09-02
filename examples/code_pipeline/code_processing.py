import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from tqdm import tqdm


def postprocess_annotations(data_path, columns):
    """
    Unifies all shards and reformats metrics into individual columns
    """

    def to_int(value):
        try:
            return int(value)
        except Exception as e:
            return pd.NA

    parquet_files = glob.glob(f"{data_path}/*.parquet")
    df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    previous_len = len(df)

    # Columns to Integers
    df[columns] = df["output"].str.split(expand=True).iloc[:, :4]
    for column in columns:
        df[column] = df[column].map(to_int).astype("Int64")

    # Post-Processing
    df = df.dropna()
    df = df[
        (df["clarity"] >= 0)
        & (df["practice"] >= 0)
        & (df["educational"] >= 0)
        & (df["difficulty"] >= 0)
    ]
    df = df[
        (df["clarity"] <= 9)
        & (df["practice"] <= 9)
        & (df["educational"] <= 9)
        & (df["difficulty"] <= 2)
    ]

    # Check sizes
    new_len = len(df)
    print(
        f"{data_path}: Removed {previous_len - new_len}, Kept: {round(previous_len/new_len, 3)}"
    )

    # Create quality (For Uniform Labeling)
    # Each score can range +-1
    # df['quality'] = pd.qcut(df[['clarity', 'practice', 'educational']].sum(axis=1) + np.random.randint(0, 4, size=len(df)), 5, labels=False, duplicates='drop')
    # columns.append('quality')

    # Check dits
    os.makedirs(f"{data_path}/figs", exist_ok=True)
    os.makedirs(f"{data_path}/postprocessed", exist_ok=True)

    # Saving figs
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, column in enumerate(columns):
        ax = axes[i // 2, i % 2]
        df[column].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="skyblue", edgecolor="black"
        )
        ax.set_title(f"{column} value counts")
        ax.set_xlabel(f"Metric {column}")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{data_path}/figs/histogram_quality.png")

    # Saving frame
    df_path = f"{data_path}/postprocessed/data.parquet"
    df.to_parquet(df_path)
    return df_path


def create_fasttext_dataset(data_path, text_column, label_column):
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
    dataset["content"] = dataset["content"].str.replace(
        "__label__", "LABEL", regex=False
    )
    fasttext_data = [
        format_to_fasttext(example[text_column], example[label_column])
        for idx, example in dataset.iterrows()
    ]
    write_to_file(output_file, fasttext_data)
    return output_file


def create_fasttext_dataset_multi(data_path, text_column, label_columns):
    """
    Creates a fasttext dataset from the given Parquet files and splits it into train/test sets.
    """

    def format_to_fasttext(text, labels):
        res = ""
        for label in labels:
            res = res + f"__label__{int(label)} "
        return res + f"{repr(text)}\n"  # [1:-1]?

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
    dataset["content"] = dataset["content"].str.replace(
        "__label__", "LABEL", regex=False
    )
    fasttext_data = [
        format_to_fasttext(example[text_column], example[label_columns].values.tolist())
        for idx, example in dataset.iterrows()
    ]
    write_to_file(output_file, fasttext_data)
    print(f"Written to: {output_file}")
    return output_file


def split_dataset_fasttext(input_file, train_ratio):
    print(input_file)
    # Read
    input_folder = os.path.dirname(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Split
    split_index = int(len(lines) * train_ratio)
    train_data = lines[:split_index]
    test_data = lines[split_index:]

    # Write
    with open(os.path.join(input_folder, "train.txt"), "w") as f:
        f.writelines(train_data)
    with open(os.path.join(input_folder, "test.txt"), "w") as f:
        f.writelines(test_data)
    return os.path.join(input_folder, "train.txt"), os.path.join(
        input_folder, "test.txt"
    )
