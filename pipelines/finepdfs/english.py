from pathlib import Path
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
import pandas as pd

if __name__ == "__main__":
    dataset_provider = "HuggingFaceFW"
    dataset_name = "finepdfs-edu"
    # Specify these directories as needed
    DATA_DIR = Path(f"./{dataset_provider}/{dataset_name}/snapshot/data/eng_Latn/train")
    OUTPUT_DIR = Path(f"./{dataset_name}-preprocessed/data")

    pipeline = [ParquetReader(str(DATA_DIR), glob_pattern="*.parquet")]

    # Robots filtering
    REMOVED_DIR = OUTPUT_DIR.parent / "removed"
    ROBOTS_PATH = "./fineweb_robots_compressed.parquet"

    robots_domains = pd.read_parquet(ROBOTS_PATH)
    robots = {row["domain"]: row["content"] for _, row in robots_domains.iterrows()}
    del robots_domains
    pipeline += [RobotsTxtFilter(robots_dict=robots,
                                 exclusion_writer=ParquetWriter(f"{REMOVED_DIR}/robots",
                                                                compression="zstd"))]
    # PII formatting
    pipeline += [PIIFormatter()]

    pipeline += [ParquetWriter(str(OUTPUT_DIR),
                               compression="zstd",
                               max_file_size=1 * 2 ** 30)]

    SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name=f"{dataset_name}_pipeline",
        logging_dir=f"{OUTPUT_DIR.parent / 'logs'}",
        tasks=len(list(DATA_DIR.rglob("*.parquet"))),
        time="03:00:00",  # Adjust time as needed
        cpus_per_task=1,
        randomize_start_duration=15,
        workers=-1,
        partition="normal",
        srun_args={
            "environment": "",
            "account": "",
        },
        sbatch_args={
            "account": "",
        },
    ).run()