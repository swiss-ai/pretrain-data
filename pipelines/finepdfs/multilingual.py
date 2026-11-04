from pathlib import Path
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
import pandas as pd

if __name__ == "__main__":
    # Specify these directories as needed
    DATA_DIR = Path(f"./HuggingFaceFW/finepdfs-edu/snapshot/data/")
    OUTPUT_DIR = Path(f"./finepdfs-edu-multilingual-preprocessed/data")

    paths_file = OUTPUT_DIR.parent / "files_path.txt"
    relative_paths = [str(path.relative_to(DATA_DIR)) for path in DATA_DIR.rglob("*.parquet") if path.parent.parent.name != 'eng_Latn']
    print(f"Found {len(relative_paths)} files to process.")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    paths_file.write_text("\n".join(str(p) for p in relative_paths))

    pipeline = [ParquetReader(str(DATA_DIR), paths_file=str(paths_file))]

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
        job_name=f"finepdfs-edu-multilingual_pipeline",
        logging_dir=f"{OUTPUT_DIR.parent / 'logs'}",
        tasks=len(relative_paths),
        time="03:00:00",  # Adjust time as needed
        cpus_per_task=6,
        randomize_start_duration=15,
        workers=8,
        partition="normal",
        srun_args={
            "environment": "",
            "account": "",
        },
        sbatch_args={
            "account": "",
        },
    ).run()