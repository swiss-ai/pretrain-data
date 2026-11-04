from pathlib import Path
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from datatrove.executor import LocalPipelineExecutor
import pandas as pd

if __name__ == "__main__":
    # Specify these directories as needed
    DATA_DIR = Path("./voilaj/swiss-caselaw")
    OUTPUT_DIR = Path("./swiss-ai/swiss-caselaw-preprocessed/data")
    REMOVED_DIR = Path("./swiss-ai/swiss-caselaw-preprocessed/removed")

    ROBOTS_PATH = "./fineweb_robots_compressed.parquet"


    # Build the processing pipeline
    # Read the files
    def reader_adapter(self, data: dict, path: str, id_in_file: int | str):
        import json
        temp = {
            "text": data.pop('full_text', ""),
            "id": data.pop('decision_id', f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": {'url': data.pop('source_url', ''),}
        }
        return temp


    pipeline = [ParquetReader(data_folder=str(DATA_DIR), adapter=reader_adapter, glob_pattern="*.parquet")]

    # Robots filtering
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
    DEBUG = False

    if DEBUG:

        LocalPipelineExecutor(pipeline=pipeline).run()

    else:
        SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name="swiss-caselaw_pipeline",
            logging_dir=f"{OUTPUT_DIR.parent / 'logs'}",
            tasks=len(list(DATA_DIR.glob("*.parquet"))),
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