from data_pipeline_pretrain.pipeline.annotators import XLMRobertaEmbeddingAnnotator
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor

DUMPS = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
    "CC-MAIN-2024-22",
    "CC-MAIN-2024-26",
    "CC-MAIN-2024-30",
    "CC-MAIN-2024-33",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-51",
]
INPUT_DIR = "./fineweb-1_3_0/snapshot/data"
OUTPUT_DIR = "./xlmroberta/fineweb-1_3_0/data"
LOGS_DIR = "./xlmroberta/fineweb-1_3_0/logs"

if __name__ == "__main__":
    last_executor = None

    for dump in DUMPS:
        pipeline = [
            ParquetReader(
                f"{INPUT_DIR}/{dump}",
                text_key="text",
            ),
            XLMRobertaEmbeddingAnnotator(
                model="FacebookAI/xlm-roberta-base",
                tokenizer_batch_size=10000,
                model_batch_size=4096,
            ),
            ParquetWriter(
                f"{OUTPUT_DIR}/{dump}",
                compression="zstd",
            ),
        ]

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"embed_fw_{dump}",
            logging_dir=f"{LOGS_DIR}/{dump}",
            tasks=256,
            workers=16,
            time="04:00:00",
            cpus_per_task=72,
            randomize_start_duration=60,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
            sbatch_args={
                "account": "",
            },
            depends=last_executor,
            run_on_dependency_fail=True,
        )

        last_executor.run()
