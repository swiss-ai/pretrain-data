from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
from data_pipeline_pretrain.pipeline.filters.toxic_filter import (
    ToxicScorer,
    ToxicityBinaryClassifierFilter,
)
from datatrove.pipeline.filters import URLFilter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
import sys
import pandas as pd


# Command-line arguments
if len(sys.argv) != 2:
    print(
        "The first argument should be the config name: python main.py quality_10-keeprobots."
    )
    exit(1)
CONFIG_NAME = sys.argv[1]

# Filters and configs
COARSE_ROBOTS_PATH = (
    "./fw1_domains.txt"
)
FINE_ROBOTS_PATH = "./fineweb-robots-txt/fineweb_robots_compressed.parquet"

CONFIGS = {
    "filterrobots_coarse": {
        "robots_filter": "coarse",
    },
    "filterrobots_fine": {
        "robots_filter": "fine",
    },
}

# Processing directories
RAW_INPUT_DIR = "./dclm-edu/snapshot/data"
BASE_OUTPUT_DIR = f"./swissai-dclm-edu-{CONFIG_NAME}"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"

if __name__ == "__main__":
    config = CONFIGS[CONFIG_NAME]

    input_dir = RAW_INPUT_DIR
    output_dir = f"{OUTPUT_DIR}/output"
    removed_dir = f"{OUTPUT_DIR}/removed"
    logs_dir = f"{LOGS_DIR}"

    pipeline = [
        ParquetReader(
            input_dir,
        ),
    ]

    if "robots_filter" in config and config["robots_filter"] == "coarse":
        print("Using coarse-grained robots filtering (domain-level)")
        with open(COARSE_ROBOTS_PATH, "r") as f:
            robots_domains = set(f.read().splitlines())
        pipeline += [
            URLFilter(
                extra_domains=robots_domains,
                exclusion_writer=ParquetWriter(
                    f"{removed_dir}/robots",
                    compression="zstd",
                ),
                use_integrated_lists=False,
            ),
        ]
    elif "robots_filter" in config and config["robots_filter"] == "fine":
        print("Using fine-grained robots filtering (URL-level)")
        robots_domains = pd.read_parquet(FINE_ROBOTS_PATH)
        robots = {row["domain"]: row["content"] for _, row in robots_domains.iterrows()}
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
        PIIFormatter(),
        ParquetWriter(
            output_dir,
            compression="zstd",
            max_file_size=1 * 2**30,  # 1GB
        ),
    ]

    print(f"Launching dclm-edu with config {config}.")
    print(f"Pipeline: {pipeline}")

    last_executor = SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name=f"dclm-edu-{CONFIG_NAME}",
        logging_dir=logs_dir,
        tasks=len(list_files(input_dir)),
        time="03:00:00",
        cpus_per_task=6,  # 4 gives OOM, likely due to models & robots
        workers=8,
        randomize_start_duration=15,
        partition="normal",
        srun_args={
            "environment": l",
            "account": "",
        },
        sbatch_args={
            "account": "",
        },
        run_on_dependency_fail=True,
    )
    last_executor.run()
