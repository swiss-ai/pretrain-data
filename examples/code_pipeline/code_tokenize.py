from data_pipeline_pretrain.executor.slurm_nodes import SlurmPipelineNodeExecutor
from datatrove.pipeline.readers import (
    ParquetReader,
)
from datatrove.pipeline.tokens import DocumentTokenizer
from data_pipeline_pretrain.utils import list_files

quantile = 0.75
INPUT_DIR = f"./dataset/starcoderdata/filtered/python/quantile_{quantile}"
OUTPUT_DIR = f"./dataset/starcoderdata/tokenized/python/quantile_{quantile}"
LOGS_DIR = f"./logs/tokenize/python/quantile_{quantile}"
LANGUAGES = ["python"]


def run_job(input_dir, output_dir, logs_dir, language, depends_executor=None):
    input_files = f"{input_dir}"
    executor = SlurmPipelineNodeExecutor(
        job_name=f"tokenize-{language}-{quantile}",
        pipeline=[
            ParquetReader(data_folder=input_files, text_key="content"),
            DocumentTokenizer(
                output_folder=f"{output_dir}",
                tokenizer_name_or_path="mistralai/Mistral-Nemo-Base-2407",
                max_tokens_per_file=1e9,
                shuffle=False,
                eos_token=None,
            ),
        ],
        tasks=len(list_files(input_files)),
        logging_dir=f"{logs_dir}",
        cpus_per_task=1,
        partition="normal",
        time="06:00:00",
        depends=depends_executor,
        srun_args={"environment": "datatrove"},
    )

    executor.run()
    return executor


if __name__ == "__main__":
    last_executor = None
    for language in LANGUAGES:
        last_executor = run_job(
            INPUT_DIR, OUTPUT_DIR, LOGS_DIR, language, last_executor
        )
