import sys
import pandas as pd

from datatrove.data import Document
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.executor.local import LocalPipelineExecutor

from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders

BASE_INPUT = ""
inputs = [
    "dpi-flan-2021/",
    "dpi-flan-dialogue/",
    "dpi-flan-sni/",
    "dpi-flan-cot/snapshot"
    ]

BASE_OUTPUT_DIR = f"./provenance-flan-templated"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs"

from typing import Callable, Literal

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger

def adapter(self, data: dict, path: str, id_in_file: int | str):
    metadata = data.pop("metadata", {})
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    input_text = data["inputs"]
    output_text = data["labels"]
    text = f"User:\n{input_text}\nAssistant:\n{output_text}"
    return {
        "text": text,
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": {},
    }

if __name__ == "__main__":
    for input_data in inputs:
        input_dir = f"{BASE_INPUT}/{input_data}"
        output_dir = f"{OUTPUT_DIR}/{input_data}/output"
        removed_dir = f"{OUTPUT_DIR}/{input_data}/removed"
        logs_dir = f"{LOGS_DIR}/{input_data}/"

        pipeline = [
            JsonlReader(
                input_dir,
                adapter=adapter,
            ),
        ]

        pipeline += [
            ParquetWriter(
                output_dir,
                compression="zstd",
                max_file_size=1 * 2**30,  # 1GB
                expand_metadata=True,
            ),
        ]

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"provenance-flan",
            logging_dir=logs_dir,
            tasks=1,
            time="03:00:00",
            cpus_per_task=6,
            workers=8,
            randomize_start_duration=15,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
            sbatch_args={
                "account": "",
            },
            run_on_dependency_fail=True,
        )
        last_executor.run()