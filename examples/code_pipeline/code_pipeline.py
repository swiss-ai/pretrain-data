import os
import glob
import argparse

from code_processing import (
    postprocess_annotations,
    create_fasttext_dataset,
    split_dataset_fasttext,
)
from code_classifier import fasttext_train, fasttext_annotate, HFClassifier
from tqdm import tqdm

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process code quality data.")
parser.add_argument("--language", type=str, help="Language to process")
# Parse arguments
args = parser.parse_args()
language = args.language

columns = ["clarity", "practice", "educational", "difficulty"]
need_classifier = [
    "c",
    "c-sharp",
    "cpp",
    "css",
    "go",
    "html",
    "java",
    "javascript",
    "json",
    "kotlin",
    "markdown",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "tex",
    "typescript",
    "yaml",
]
print(language)
DATASET_PATH = "./quality"
print("POSTPROCESS")
df_postprocessed_path = postprocess_annotations(f"{DATASET_PATH}/{language}", columns)
if language in need_classifier:
    for column in columns:
        df_path = create_fasttext_dataset(df_postprocessed_path, "content", column)
        split_dataset_fasttext(df_path, 0.9)
    print("TRAIN")
    fasttext_train(language, columns)
    print("ANNOTATE")
    fasttext_annotate(language, columns)
