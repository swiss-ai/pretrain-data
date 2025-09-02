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
        
        text_info = data.pop(self.text_key, [])

        def generate_conversation(chat_log):
            conversation = []
            for entry in chat_log:
                speaker = entry["from"]
                text = entry["value"]
                if speaker == "human":
                    conversation.append(f"user: {text}")
                else:
                    conversation.append(f"assistant: {text}")
            return "\n".join(conversation)

        text = generate_conversation(text_info)
        return {
            "text": text,
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": {},
        }

# Processing directories
RAW_INPUT_DIR = "./euroblocks/snapshot/data/"
BASE_OUTPUT_DIR = f"./euroblocks-templated"
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
            text_key="conversations",
            adapter=adapter,
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
    print(f"Pipeline: {pipeline}, {list_files(input_dir)}, {input_dir}")
    last_executor = SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name=f"euroblock",
        logging_dir=logs_dir,
        tasks=2,
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
