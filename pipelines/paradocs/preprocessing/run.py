import os
import gzip
import itertools
from pathlib import Path
import subprocess
import shlex
from datetime import datetime
import time

def find_lang_documents_values(base_path: Path):
    languages = {}

    if not base_path.is_dir():
        print(f"Error: Base directory '{base_path}' not found.")
        return {}

    for language_dir in base_path.iterdir():
        if language_dir.is_dir():
            language = language_dir.name
            strict_path = language_dir / "strict"
            fallback_path = language_dir / "all/paracrawl"
            
            scan_path = None
            if strict_path.is_dir():
                scan_path = strict_path
                level = "strict"
            elif fallback_path.is_dir():
                scan_path = fallback_path
                level = "all"

            if scan_path and scan_path.is_dir():
                documents = []
                for document_file in scan_path.glob("*.gz"):
                    documents.append((document_file, document_file.stem, level))
                languages[language] = documents
    return languages

def submit_job(language: str, name: str, level: str, file_path: str):
    print(f"Submitting job for path: {file_path}")
    languages = language.split("-")
    lang0 = languages[0]
    lang1 = languages[1]
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_output_filename = f"./paradocs-bidirectional/logs/{language}/{level}/{name}/out-{timestamp_str}.log"
    ratio = 1.0 if level == "strict" else 0.25
    python_command = f"cd pipelines/paradocs/preprocessing && python3 paradocs.py --input {file_path} --src {lang0} --tgt {lang1} --frequency_cutoff 3 --sample_ratio {ratio} --output ./paradocs-bidirectional/data/{language}/{level}/{name}.parquet"
    sbatch_command = [
        'sbatch',
        '--account', '',
        '--job-name', f"{language}-paradocs",
        '--environment', "",
        '-o', log_output_filename,
        '--time', '03:30:00',
        '--wrap', python_command
    ]
    print(f"Executing command:\n{' '.join(shlex.quote(arg) for arg in sbatch_command)}")

    result = subprocess.run(
            sbatch_command,
        )
    
def main():
    base_path = Path("./paradocs/snapshot/data/")
    lang_documents = find_lang_documents_values(base_path)
    for lang, documents in lang_documents.items():
        for file_path, document_name, level in documents:
            if file_path.exists():
                submit_job(lang, document_name, level, file_path)
                time.sleep(3)

if __name__ == "__main__":
    main()
