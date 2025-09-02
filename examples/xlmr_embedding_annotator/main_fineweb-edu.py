from data_pipeline_pretrain.pipeline.annotators import XLMRobertaEmbeddingAnnotator
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor

DUMPS = [
    "CC-MAIN-2013-20",
    "CC-MAIN-2014-52",
    "CC-MAIN-2015-40",
    "CC-MAIN-2016-44",
    "CC-MAIN-2017-34",
    "CC-MAIN-2018-22",
    "CC-MAIN-2019-09",
    "CC-MAIN-2019-47",
    "CC-MAIN-2020-45",
    "CC-MAIN-2021-43",
    "CC-MAIN-2023-14",
    "CC-MAIN-2013-48",
    "CC-MAIN-2015-06",
    "CC-MAIN-2015-48",
    "CC-MAIN-2016-50",
    "CC-MAIN-2017-39",
    "CC-MAIN-2018-26",
    "CC-MAIN-2019-13",
    "CC-MAIN-2019-51",
    "CC-MAIN-2020-50",
    "CC-MAIN-2021-49",
    "CC-MAIN-2023-23",
    "CC-MAIN-2014-10",
    "CC-MAIN-2015-11",
    "CC-MAIN-2016-07",
    "CC-MAIN-2017-04",
    "CC-MAIN-2017-43",
    "CC-MAIN-2018-30",
    "CC-MAIN-2019-18",
    "CC-MAIN-2020-05",
    "CC-MAIN-2021-04",
    "CC-MAIN-2022-05",
    "CC-MAIN-2023-40",
    "CC-MAIN-2014-15",
    "CC-MAIN-2015-14",
    "CC-MAIN-2016-18",
    "CC-MAIN-2017-09",
    "CC-MAIN-2017-47",
    "CC-MAIN-2018-34",
    "CC-MAIN-2019-22",
    "CC-MAIN-2020-10",
    "CC-MAIN-2021-10",
    "CC-MAIN-2022-21",
    "CC-MAIN-2023-50",
    "CC-MAIN-2014-23",
    "CC-MAIN-2015-18",
    "CC-MAIN-2016-22",
    "CC-MAIN-2017-13",
    "CC-MAIN-2017-51",
    "CC-MAIN-2018-39",
    "CC-MAIN-2019-26",
    "CC-MAIN-2020-16",
    "CC-MAIN-2021-17",
    "CC-MAIN-2022-27",
    "CC-MAIN-2024-10",
    "CC-MAIN-2014-35",
    "CC-MAIN-2015-22",
    "CC-MAIN-2016-26",
    "CC-MAIN-2017-17",
    "CC-MAIN-2018-05",
    "CC-MAIN-2018-43",
    "CC-MAIN-2019-30",
    "CC-MAIN-2020-24",
    "CC-MAIN-2021-21",
    "CC-MAIN-2022-33",
    "CC-MAIN-2014-41",
    "CC-MAIN-2015-27",
    "CC-MAIN-2016-30",
    "CC-MAIN-2017-22",
    "CC-MAIN-2018-09",
    "CC-MAIN-2018-47",
    "CC-MAIN-2019-35",
    "CC-MAIN-2020-29",
    "CC-MAIN-2021-25",
    "CC-MAIN-2022-40",
    "CC-MAIN-2014-42",
    "CC-MAIN-2015-32",
    "CC-MAIN-2016-36",
    "CC-MAIN-2017-26",
    "CC-MAIN-2018-13",
    "CC-MAIN-2018-51",
    "CC-MAIN-2019-39",
    "CC-MAIN-2020-34",
    "CC-MAIN-2021-31",
    "CC-MAIN-2022-49",
    "CC-MAIN-2014-49",
    "CC-MAIN-2015-35",
    "CC-MAIN-2016-40",
    "CC-MAIN-2017-30",
    "CC-MAIN-2018-17",
    "CC-MAIN-2019-04",
    "CC-MAIN-2019-43",
    "CC-MAIN-2020-40",
    "CC-MAIN-2021-39",
    "CC-MAIN-2023-06",
]
INPUT_DIR = "./fineweb-edu-full/data/"
OUTPUT_DIR = "./mfineweb-draft/embeddings/xlmroberta/fineweb-edu-full/"
LOGS_DIR = "./logs/embeddings/xlmroberta/fineweb-edu-full/"

if __name__ == "__main__":
    last_executor = None

    for dump in DUMPS:
        pipeline = [
            ParquetReader(
                f"{INPUT_DIR}/{dump}/",
            ),
            XLMRobertaEmbeddingAnnotator(
                model="FacebookAI/xlm-roberta-base",
                tokenizer_batch_size=10000,
                model_batch_size=4096,
            ),
            ParquetWriter(
                f"{OUTPUT_DIR}/{dump}/",
                compression="zstd",
            ),
        ]

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"data_fw-edu_{dump}",
            logging_dir=f"{LOGS_DIR}/{dump}/",
            tasks=32,
            workers=1,
            time="03:00:00",
            cpus_per_task=72,  # -> 4 tasks (=4GPUs) per node
            randomize_start_duration=30,
            partition="normal",
            srun_args={
                "environment": ""
            },
            # depends=last_executor,
            run_on_dependency_fail=True,
        )

        last_executor.run()
