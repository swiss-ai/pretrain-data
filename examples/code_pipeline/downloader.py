import os
import argparse
import gzip
import shutil
import glob

from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_stackv1(workers, data_dir, hf_dir, token):
    os.environ['HF_HOME'] = hf_dir
    os.environ['HTTP_PROXY'] = "http://proxy.cscs.ch:8080"
    os.environ['HTTPS_PROXY'] = "http://proxy.cscs.ch:8080"
    os.environ['NO_PROXY'] = ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"

    snapshot_download(
        "bigcode/starcoderdata",
        repo_type="dataset",
        local_dir=data_dir,
        max_workers=workers,
        token=token
    )

def download_stackv2(workers, data_dir, hf_dir, token):
    os.environ['HF_HOME'] = hf_dir
    os.environ['HTTP_PROXY'] = "http://proxy.cscs.ch:8080"
    os.environ['HTTPS_PROXY'] = "http://proxy.cscs.ch:8080"
    os.environ['NO_PROXY'] = ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"

    snapshot_download(
        "common-pile/stackv2_edu_filtered",
        repo_type="dataset",
        local_dir=os.path.join(data_dir, "edu"),
        max_workers=workers,
        token=token
    )
    snapshot_download(
        "common-pile/stackv2_html_filtered",
        repo_type="dataset",
        local_dir=os.path.join(data_dir, "html"),
        max_workers=workers,
        token=token
    )

    # Convert to jsonl
    def decompress_and_delete(gz_path):
        output_file = gz_path.with_suffix('')
        with gzip.open(gz_path, 'rb') as f_in, open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        return str(gz_path)

    # Convert to jsonl
    print("Converting to JSON", flush=True)
    input_dir = Path(os.path.join(data_dir, "edu"))
    gz_files = list(input_dir.glob("*.json.gz"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(decompress_and_delete, gz) for gz in gz_files]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error: {e}")
    
    # Now create dataset based on the language
    print("Filtering by language", flush=True)
    dataset = load_dataset('json', data_files=os.path.join(input_dir, "*.json"), num_proc=workers)['train']
    dataset = dataset.map(lambda example: {**example, 'language': example['metadata']['language']}, num_proc=workers)
    languages = set(dataset.unique('language'))
    for language in languages:
        language_dataset = dataset.filter(lambda example: example['language'] == language, num_proc=workers)
        language_dataset.to_parquet(os.path.join(input_dir, f'{language}.parquet'))

    # Clean up
    for file_path in glob.glob(os.path.join(input_dir, "*.json")):
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download code datasets.")
    parser.add_argument("--dataset", type=str, choices=['stackv1', 'stackv2'], required=True, help="Dataset to download: 'stack' or 'extras'")
    parser.add_argument("--workers", type=int, default=164, help="Number of workers to use for downloading.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to store downloaded data.")
    parser.add_argument("--token", type=str, required=False, help="Hugging Face token.")
    parser.add_argument("--hf_dir", type=str, required=False, help="Hugging Face cache directory.")

    args = parser.parse_args()

    if args.dataset == "stackv1":
        download_stackv1(args.workers, args.data_dir, args.hf_dir, args.token)
    if args.dataset == "stackv2":
        download_stackv2(args.workers, args.data_dir, args.hf_dir, args.token)
    else:
        raise "Invalid choice to download data"