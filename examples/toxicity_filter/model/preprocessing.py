import torch
import numpy as np
import pandas as pd
import argparse


def get_toxic_score(x):
    total_score = np.sum([int(a.strip()) for a in x[1:-1].split(",")])
    if total_score > 0:
        return 1
    else:
        return 0


def process_dataset(raw_path, save_path=None):
    if save_path is None:
        save_path = raw_path.replace("annotate", "processed")

    if isinstance(raw_path, list):
        if len(raw_path) > 1:
            raw_df = pd.concat([pd.read_csv(path) for path in raw_path])
        else:
            raw_df = pd.read_csv(raw_path[0])
    else:
        raw_df = pd.read_csv(raw_path)

    raw_df = raw_df[
        raw_df["scores"].apply(lambda x: x.startswith("[") and x.endswith("]"))
    ]
    raw_df = raw_df[raw_df["original_text"].apply(lambda x: isinstance(x, str))]
    raw_df["scores"] = raw_df["scores"].apply(lambda x: get_toxic_score(x))

    processed_df = raw_df[["original_text", "scores"]]
    processed_df.rename(
        columns={"original_text": "content", "scores": "toxic"}, inplace=True
    )
    toxic_df = processed_df[processed_df["toxic"] == 1]
    nontoxic_df = processed_df[processed_df["toxic"] == 0]
    subsampled_nontoxic_df = nontoxic_df.sample(n=min(len(toxic_df), len(nontoxic_df)))
    all_df = pd.concat([toxic_df, subsampled_nontoxic_df]).sample(frac=1)
    all_df = all_df.sample(frac=1)
    all_df = all_df.sample(frac=1)
    all_df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    return all_df


args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    "--data_dir", default="./swiss-ai/data/PleIAs", type=str
)
args_parser.add_argument(
    "--processed_dir",
    default="./swiss-ai/data/PleIAs_processed/",
    type=str,
)

if __name__ == "__main__":
    args = args_parser.parse_args()
    DATA_DIR = args.data_dir
    PROCESSED_DIR = args.processed_dir
    LANG2DATAPATH = {
        "dutch": [f"{DATA_DIR}/annotated_Dutch-PD.csv"],
        "french": [
            f"{DATA_DIR}/annotated_French-PD-Books.csv",
            f"{DATA_DIR}/annotated_French-PD-Newspapers.csv",
        ],
        "german": [
            f"{DATA_DIR}/annotated_German-PD.csv",
            f"{DATA_DIR}/annotated_German-PD-Newspapers.csv",
        ],
        "italian": [f"{DATA_DIR}/annotated_Italian-PD.csv"],
        "polish": [f"{DATA_DIR}/annotated_Polish-PD.csv"],
        "spanish": [
            f"{DATA_DIR}/annotated_Spanish-PD-Books.csv",
            f"{DATA_DIR}/annotated_Spanish-PD-Newspapers.csv",
        ],
        "portuguese": [
            f"{DATA_DIR}/annotated_Portuguese-PD.csv",
        ],
        "english": [
            f"{DATA_DIR}/annotated_US-PD-Books.csv",
            f"{DATA_DIR}/annotated_US-PD-Newspapers.csv",
        ],
    }

    LANG2ProcessPATH = {
        "dutch": f"{PROCESSED_DIR}/Dutch_processed.csv",
        "french": f"{PROCESSED_DIR}/French_processed.csv",
        "german": f"{PROCESSED_DIR}/German_processed.csv",
        "italian": f"{PROCESSED_DIR}/Italian_processed.csv",
        "polish": f"{PROCESSED_DIR}/Polish_processed.csv",
        "spanish": f"{PROCESSED_DIR}/Spanish_processed.csv",
        "portuguese": f"{PROCESSED_DIR}/Portuguese_processed.csv",
        "english": f"{PROCESSED_DIR}/English_processed.csv",
    }

    for lang, datapaths in LANG2DATAPATH.items():
        df = process_dataset(datapaths, save_path=LANG2ProcessPATH[lang])
        print(f"Length of {lang} dataset: {len(df)}")
