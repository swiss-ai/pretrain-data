import os
import json

from collections import defaultdict
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from glob import glob


def process_file(file):
    language = os.path.splitext(os.path.basename(file))[0]
    dataset = load_dataset('parquet', data_files=file)['train']
    return language, {
        "size": os.path.getsize(file),
        "len": len(dataset),
        "tokens": -1
    }


def root(datapath, output_file="dataset_stats.json", max_workers=128):
    results = {}

    for folder in glob(datapath):
        threshold = os.path.basename(folder.rstrip("/"))
        results[threshold] = {}

        parquet_files = glob(os.path.join(folder, "*.parquet"))

        # Run in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(process_file, parquet_files)

        for language, stats in futures:
            results[threshold][language] = stats

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved stats to {output_file}")



def json_to_metric_tables_latex_absolute(json_path):
    """
    Converts nested threshold JSON into LaTeX longtables
    for each metric: size, len, and tokens.
    - Size shown in GB with 2 decimals
    - Len and Tokens shown as integers
    - Tokens with value -1 shown as -1
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    thresholds = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    languages = sorted(set(lang for d in data.values() for lang in d))

    # Organize raw data
    metrics = {"size": defaultdict(dict), "len": defaultdict(dict), "tokens": defaultdict(dict)}
    for t in thresholds:
        for lang in languages:
            stats = data[t].get(lang, {})
            for m in metrics:
                metrics[m][lang][t] = stats.get(m, None)

    def format_value(metric, value):
        if value is None:
            return "N/A"
        if metric == "tokens" and value == -1:
            return "-1"
        if metric == "size":
            return f"{value / 1_000_000_000:.2f}~GB"
        return str(value)

    def build_table(metric):
        column_format = "|l|" + "c|" * len(thresholds)
        header = "\\begin{longtable}{" + column_format + "}\n"
        header += "\\hline\n"
        header += "\\textbf{Language} & " + " & ".join([f"\\textbf{{{t}}}" for t in thresholds]) + " \\\\\n"
        header += "\\hline\n"
        header += "\\endfirsthead\n"
        header += "\\hline\n"
        header += "\\textbf{Language} & " + " & ".join([f"\\textbf{{{t}}}" for t in thresholds]) + " \\\\\n"
        header += "\\hline\n"
        header += "\\endhead\n"
        header += "\\hline\n"
        header += "\\endfoot\n"
        header += "\\hline\n"
        header += "\\endlastfoot\n"

        rows = ""
        for lang in languages:
            row = [lang]
            for t in thresholds:
                val = metrics[metric][lang].get(t)
                row.append(format_value(metric, val))
            rows += " & ".join(row) + r" \\" + "\n"

        footer = "\\end{longtable}"
        caption = f"\\textbf{{Metric: {metric.capitalize()}}}"
        return caption + "\n\n" + header + rows + footer

    return "\n\n".join([build_table(m) for m in ["size", "len", "tokens"]])


def extract_tokens(root):
    results = {}
    for folder in glob(root):
        basename = os.path.basename(folder)
        results[basename] = {}
        total_tokens = 0
        for dump in glob(f'{folder}/*'):
            with open(os.path.join(dump, 'stats.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            dump_tokens = data[1]["stats"]["tokens"]["total"]
            total_tokens += dump_tokens
        results[basename] = {"tokens": total_tokens}
    return results