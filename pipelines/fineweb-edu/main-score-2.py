from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import IdFilter, load_robots
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

import sys

# Command-line arguments
if len(sys.argv) != 2:
    print(
        "The first argument should be the config name: python main-score-2.py keeprobots."
    )
    exit(1)
CONFIG_NAME = sys.argv[1]

# Dumps to process
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

CONFIGS = {
    "filterrobots": {
        "filter_robots": True,
    },
    "keeprobots": {
        "filter_robots": False,
    },
}

INPUT_DIR = (
    "./fineweb-edu-score-2/snapshot/data/"
)
BASE_OUTPUT_DIR = (
    f"./swissai-fineweb-edu-score-2-{CONFIG_NAME}/"
)
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"

ROBOTS_DIR = "./robotstxt/new_compute_permissivity/fw1_permissivity_logs"

if __name__ == "__main__":
    config = CONFIGS[CONFIG_NAME]

    for dump in DUMPS:
        input_dir = f"{INPUT_DIR}/{dump}"
        output_dir = f"{OUTPUT_DIR}/output/{dump}"
        removed_dir = f"{OUTPUT_DIR}/removed/{dump}"
        logs_dir = f"{LOGS_DIR}/{dump}/"

        pipeline = [
            ParquetReader(
                input_dir,
            ),
        ]

        if "filter_robots" in config and config["filter_robots"] is True:
            pipeline += [
                IdFilter(
                    ids_to_filter=load_robots(f"{ROBOTS_DIR}/{dump}"),
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots", compression="zstd"
                    ),
                )
            ]

        pipeline += [
            PIIFormatter(
                add_pii_list_to_metadata=False,
            ),
            ParquetWriter(
                output_dir,
                compression="zstd",
                max_file_size=1 * 2**30,  # 1GB
            ),
        ]

        SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"fwedu-score2-{dump}-{CONFIG_NAME}",
            logging_dir=logs_dir,
            tasks=len(list_files(input_dir)),
            time="08:00:00",
            cpus_per_task=6,
            randomize_start_duration=10,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
        ).run()
