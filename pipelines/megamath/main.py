from datatrove.pipeline.filters import URLFilter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
import sys
import pandas as pd

# Command-line arguments
if len(sys.argv) != 2:
    print("The first argument should be the config name: python main.py filterrobots")
    exit(1)
CONFIG_NAME = sys.argv[1]


# Subsets to process
SUBSETS = [
    "megamath-web",
    "megamath-web-pro",
]

INPUT_DIR = "./LLM360-MegaMath/snapshot"
BASE_OUTPUT_DIR = f"./swissai-megamath-{CONFIG_NAME}/"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"
FINE_ROBOTS_PATH = "./fineweb-robots-txt/fineweb_robots_compressed.parquet"

CONFIGS = {
    "filterrobots": {
        "robots_filter": True,
    },
}

if __name__ == "__main__":
    config = CONFIGS[CONFIG_NAME]

    for subset in SUBSETS:
        input_dir = f"{INPUT_DIR}/{subset}"
        output_dir = f"{OUTPUT_DIR}/output/{subset}"
        removed_dir = f"{OUTPUT_DIR}/removed/{subset}"
        logs_dir = f"{LOGS_DIR}/{subset}/"

        pipeline = [
            ParquetReader(
                input_dir,
            ),
        ]

        if "robots_filter" in config and config["robots_filter"] is True:
            print("Using fine-grained robots filtering (URL-level)")
            robots_domains = pd.read_parquet(FINE_ROBOTS_PATH)
            robots = {
                row["domain"]: row["content"] for _, row in robots_domains.iterrows()
            }
            del robots_domains
            pipeline += [
                RobotsTxtFilter(
                    robots_dict=robots,
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots",
                        compression="zstd",
                    ),
                ),
            ]

        pipeline += [
            ParquetWriter(
                output_dir,
                compression="zstd",
                max_file_size=1 * 2**30,  # 1GB
            ),
        ]

        print(f"Launching megamath with config {config}.")
        print(f"Pipeline: {pipeline}")

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"megamath-{CONFIG_NAME}-{subset}",
            logging_dir=logs_dir,
            tasks=max(1, len(list_files(input_dir, recursive=True)) // 20),
            time="08:00:00",
            cpus_per_task=4,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
            sbatch_args={
                "account": "",
            },
        )
        last_executor.run()
