import sys
import pandas as pd

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor

from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders

# Processing directories
RAW_INPUT_DIR = "./paradocs-bidirectional/data"
BASE_OUTPUT_DIR = f"./paradocs-bidirectional-piifiltered/"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs"

# Languages to process
LANGUAGES = [
    folder.name
    for folder in list_folders(RAW_INPUT_DIR)
]

if __name__ == "__main__":
    for language in LANGUAGES:
        input_dir = f"{RAW_INPUT_DIR}/{language}"
        output_dir = f"{OUTPUT_DIR}/{language}/output"
        removed_dir = f"{OUTPUT_DIR}/{language}/removed"
        logs_dir = f"{LOGS_DIR}/{language}/"

        pipeline = [
            ParquetReader(
                input_dir,
            ),
        ]

        pipeline += [
            PIIFormatter(),
            ParquetWriter(
                output_dir,
                compression="zstd",
                max_file_size=1 * 2**30,  # 1GB
            ),
        ]

        print(f"Launching paradocs for language {language}.")
        print(f"Pipeline: {pipeline}")

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"{language}-paradocs",
            logging_dir=logs_dir,
            tasks=5,
            time="03:00:00",
            cpus_per_task=6,  # 4 gives OOM, likely due to models & robots
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