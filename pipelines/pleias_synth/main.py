import sys
import pandas as pd

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor

from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders

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
        
        def generate_conversation(entry):
            conversation = []
            query = entry["query"]
            synthetic_reasoning = entry["synthetic_reasoning"]
            synthetic_answer = entry["synthetic_answer"]
            conversation.append(f"User: {query}")
            conversation.append(f"Assistant: {synthetic_answer}")
            return "\n".join(conversation)

        text = generate_conversation(data)
        return {
            "text": text,
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": {},
        }

RAW_INPUT_DIR = "./PleIAs/SYNTH/"
BASE_OUTPUT_DIR = f"./SYNTH-preprocessed/"
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
    ]
    pipeline += [
        PIIFormatter(),
        ParquetWriter(
            output_dir,
            compression="zstd",
            max_file_size=1 * 2**30,
        ),
    ]
    print(f"Pipeline: {pipeline}, {list_files(input_dir)}, {input_dir}")
    last_executor = SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name=f"SYNTH_pipeline",
        logging_dir=logs_dir,
        tasks=50,
        time="05:59:00",
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