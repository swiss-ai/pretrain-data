from data_pipeline_pretrain.pipeline.annotators import XLMRobertaEmbeddingAnnotator
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files

LANGUAGES = [
    "deu_Latn",
    "fra_Latn",
    "cmn_Hani",
    "rus_Cyrl",
    "jpn_Jpan",
    "spa_Latn",
    "ita_Latn",
    "por_Latn",
    "pol_Latn",
    "nld_Latn",
    "ind_Latn",
    "tur_Latn",
    "ces_Latn",
    "vie_Latn",
    "swe_Latn",
    "fas_Arab",
    "arb_Arab",
    "ell_Grek",
    "dan_Latn",
    "hun_Latn",
]
INPUT_DIR = "./fineweb-2_0_1/snapshot/data/"
OUTPUT_DIR = "./mfineweb-draft/embeddings/xlmroberta/fineweb-2_0_1/"
LOGS_DIR = "./embeddings/xlmroberta/fineweb-2_0_1/"

if __name__ == "__main__":
    last_executor = None
    for lang in LANGUAGES:
        pipeline = [
            ParquetReader(
                f"{INPUT_DIR}/{lang}/train",
            ),
            XLMRobertaEmbeddingAnnotator(
                model="FacebookAI/xlm-roberta-base",
                tokenizer_batch_size=10000,
                model_batch_size=4096,
            ),
            ParquetWriter(
                f"{OUTPUT_DIR}/{lang}/",
                compression="zstd",
            ),
        ]

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"data_fw-201_{lang}",
            logging_dir=f"{LOGS_DIR}/{lang}/",
            tasks=min(len(list_files(f"{INPUT_DIR}/{lang}/train")), 400),
            workers=100,
            time="06:00:00",
            cpus_per_task=72,  # -> 4 tasks (=4GPUs) per node
            randomize_start_duration=30,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
            depends=last_executor,
            run_on_dependency_fail=True,
        )

        last_executor.run()
