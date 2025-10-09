from datatrove.pipeline.readers import (
    ParquetReader,
    JsonlReader
)
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer
from data_pipeline_pretrain.utils import list_files
from data_pipeline_pretrain.pipeline.filters.code_quality_filter import CodeQualityThresholdFilter
from data_pipeline_pretrain.executor.slurm_nodes import SlurmPipelineNodeExecutor


def run_job(input_dir, output_dir, logs_dir, depends_executor=None):
    input_files = f"{input_dir}"
    executor = SlurmPipelineNodeExecutor(
        job_name=f"tokenize",
        pipeline=[
            JsonlReader(data_folder=input_files, text_key='content'),
            MegatronDocumentTokenizer(
                output_folder=f"{output_dir}",
                tokenizer_name_or_path="alehc/swissai-tokenizer",
                max_tokens_per_file=1e9,
                shuffle=False,
                bos_token="<s>",
                eos_token="</s>",
            ),
        ],
        tasks=len(list_files(input_files)),
        logging_dir=f"{logs_dir}",
        cpus_per_task=1,
        partition="normal",
        time="06:00:00",
        depends=depends_executor,
        srun_args={
            "environment": "datatrove"
        },
    )

    executor.run()
    return executor


if __name__ == "__main__":
    last_executor = None

    for threshold in [0, 1, 2, 3, 4]:
        INPUT_DIR = f"/iopsstor/scratch/cscs/rmachace/dataset/stackv2/edu-threshold/threshold_{threshold}"
        OUTPUT_DIR = f"/iopsstor/scratch/cscs/rmachace/dataset/starcoderdata/tokenized/stackv2_edu/threshold_{threshold}"
        LOGS_DIR = f"/iopsstor/scratch/cscs/rmachace/logs/tokenize/stackv2_threshold_{threshold}"
        last_executor = run_job(INPUT_DIR, OUTPUT_DIR, LOGS_DIR, last_executor)