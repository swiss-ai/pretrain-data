import sys
import json
import random
import argparse

from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
from datatrove.pipeline.readers import ParquetReader
from data_pipeline_pretrain.pipeline.filters import IdFilter, load_robots
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.executor.local import LocalPipelineExecutor

from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files

ROBOTS_DIR = "./fineweb_robots_compressed.parquet"


class TextPermuter(PipelineStep):
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        import random
        for doc in data:
            lang_name = doc.metadata.get("og_language", "original")

            en_part = "en: " + doc.text
            og_part = f"{lang_name}: " + doc.metadata.get("og_full_text", "")

            parts = [en_part, og_part]
            random.shuffle(parts)

            doc.text = "\n".join(parts)
            yield doc


def adapter(self, data: dict, path: str, id_in_file: int | str):
    metadata = data.pop("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}

    metadata["og_full_text"] = data["og_full_text"]
    metadata["og_language"] = data["og_language"]

    score = data.get("edu_score_raw", 0.0)
    metadata["edu_score_raw"] = float(score) if score is not None else 0.0

    return {
        "text": data["translated_text"],
        "id": data.pop("id", f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": metadata,
    }


RAW_INPUT_DIR = "./HuggingFaceFW/finetranslations/data/"
BASE_OUTPUT_DIR = f"./finetranslations_preprocessed"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs"

if __name__ == "__main__":
    input_dir = f"{RAW_INPUT_DIR}"
    output_dir = f"{OUTPUT_DIR}/output"
    removed_dir = f"{OUTPUT_DIR}/removed"
    logs_dir = f"{LOGS_DIR}/"

    pipeline = [
        ParquetReader(
            input_dir,
            adapter=adapter,
            glob_pattern="**/*.parquet"
        ),
        IdFilter(
            ids_to_filter=load_robots(f"{ROBOTS_DIR}"),
            exclusion_writer=ParquetWriter(
                f"{removed_dir}/robots", compression="zstd"
            ),
        ),
        LambdaFilter(
            lambda doc: doc.metadata.get("edu_score_raw", 0.0) > 2.5
        ),
        TextPermuter(),
    ]

    pipeline += [
        PIIFormatter(),
        ParquetWriter(
            output_dir,
            compression="zstd",
            max_file_size=1 * 2 ** 30,
        ),
    ]

    executor = SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name=f"finetranslations",
        logging_dir=logs_dir,
        tasks=50,
        time="11:59:00",
        cpus_per_task=4,
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

    executor.run()