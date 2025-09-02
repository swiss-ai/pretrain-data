import os
import datasets
import glob
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


def filter_data(data_path, output_path, quantile):
    final_path = os.path.join(output_path, f"quantile_{quantile}")
    os.makedirs(final_path, exist_ok=True)

    def to_int(value):
        try:
            return int(value)
        except Exception as e:
            return pd.NA

    dataset = pd.read_parquet(data_path)
    full_length = len(dataset)
    for column in columns:
        dataset[column] = dataset[column].map(to_int).astype("Int64")
    dataset = dataset.dropna()

    # Filter
    thresholds = dataset[columns].quantile(quantile)
    selection = dataset[columns].ge(thresholds).all(axis=1)
    dataset = dataset[selection]
    new_length = len(dataset)
    print(f"{data_path}: {new_length/full_length}")

    # Get the directory path
    directory = os.path.dirname(data_path)
    file_name = os.path.basename(data_path)
    dataset.to_parquet(os.path.join(final_path, file_name))

    # Saving figs
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, column in enumerate(columns):
        ax = axes[i // 2, i % 2]
        dataset[column].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="skyblue", edgecolor="black"
        )
        ax.set_title(f"{column} value counts")
        ax.set_xlabel(f"Metric {column}")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{final_path}/histogram_quality.png")
    plt.close()


columns = ["clarity", "practice", "educational"]
quantile = 0
for file in tqdm(
    glob.glob(f"./datasets/python/*.parquet")
):
    filter_data(
        file,
        "./dataset/starcoderdata/filtered/python",
        quantile,
    )
