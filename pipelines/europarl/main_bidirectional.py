import sys
import pandas as pd

from functools import partial

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor

from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders

# Processing directories
RAW_INPUT_DIR = "./europarl-helsinki-nlp/snapshot"
BASE_OUTPUT_DIR = f"./swissai-europarl-helsinki-nlp-bidirectional-preprocessed"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs"

# Languages to process
LANGUAGES = [
    folder.name
    for folder in list_folders(RAW_INPUT_DIR)
]

if __name__ == "__main__":
    for language in LANGUAGES:
        src_lang = language.split("-")[0]
        dst_lang = language.split("-")[1]

        for is_reverse in [ True, False ]:
            def adapter(self, data: dict, path: str, id_in_file: int | str, is_reverse: bool):
                metadata = data.pop("metadata", {})
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        pass
                if not isinstance(metadata, dict):
                    metadata = {"metadata": metadata}
                
                text_info = data.pop(self.text_key, {})
                (lang1, text1), (lang2, text2) = text_info.items()
                if not is_reverse:
                    text = f"{lang1}: {text1}\n{lang2}: {text2}"
                else:
                    text = f"{lang2}: {text2}\n{lang1}: {text1}"
                return {
                    "text": text,
                    "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                    "media": data.pop("media", []),
                    "metadata": metadata | data,
                }
        
            input_dir = f"{RAW_INPUT_DIR}/{language}"
            output_dir = f"{OUTPUT_DIR}/{language}/is_reverse_{is_reverse}/output"
            removed_dir = f"{OUTPUT_DIR}/{language}/is_reverse_{is_reverse}/removed"
            logs_dir = f"{LOGS_DIR}/{language}/is_reverse_{is_reverse}"

            pipeline = [
                ParquetReader(
                    input_dir,
                    text_key="translation",
                    adapter=partial(adapter, is_reverse=is_reverse),
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

            print(f"Launching europarl for language {language}.")
            print(f"Pipeline: {pipeline}")

            last_executor = SlurmPipelineNodeExecutor(
                pipeline=pipeline,
                job_name=f"{language}-europarl",
                logging_dir=logs_dir,
                tasks=len(list_files(input_dir)),
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