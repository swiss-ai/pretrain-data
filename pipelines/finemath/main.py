from datatrove.pipeline.filters import URLFilter
from datatrove.executor.local import LocalPipelineExecutor
from data_pipeline_pretrain.utils import list_files
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.pipeline.filters import RobotsTxtFilter
import sys
import pandas as pd

# Command-line arguments
if len(sys.argv) != 2:
    print(
        "The first argument should be the config name: python main.py filterrobots_fine."
    )
    exit(1)
CONFIG_NAME = sys.argv[1]


# Subsets to process
SUBSETS = [
    "finemath-3plus",
    "finemath-4plus",
    "infiwebmath-3plus",
    "infiwebmath-4plus",
]

INPUT_DIR = "./finemath/snapshot/"
BASE_OUTPUT_DIR = f"./swissai-finemath-{CONFIG_NAME}/"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"
COARSE_ROBOTS_PATH = (
    "./robots/fw1_domains.txt"
)
FINE_ROBOTS_PATH = "./fineweb-robots-txt/fineweb_robots_compressed.parquet"

CONFIGS = {
    "filterrobots": {
        "robots_filter": "coarse",
    },
    "filterrobots_fine": {
        "robots_filter": "fine",
    },
}


def finemath_reader_adapter(self, data: dict, path: str, id_in_file: int | str):
    """
    The default data adapter to adapt input data into the datatrove Document format

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    import json

    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": json.loads(data.pop("metadata", "{}"))
        | data,  # remaining data goes into metadata
    }


def finemath_writer_adapter(self, document) -> dict:
    import json
    import dataclasses

    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    data["metadata"] = json.dumps(data["metadata"])
    return data


if __name__ == "__main__":
    config = CONFIGS[CONFIG_NAME]

    for subset in SUBSETS:
        input_dir = f"{INPUT_DIR}/{subset}"
        output_dir = f"{OUTPUT_DIR}/output/{subset}"
        removed_dir = f"{OUTPUT_DIR}/removed/{subset}"
        logs_dir = f"{LOGS_DIR}/{subset}/"

        pipeline = [
            ParquetReader(
                input_dir,
                adapter=finemath_reader_adapter,
            ),
        ]

        if "robots_filter" in config and config["robots_filter"] == "coarse":
            print("Using coarse-grained robots filtering (domain-level)")
            with open(COARSE_ROBOTS_PATH, "r") as f:
                robots_domains = set(f.read().splitlines())
            pipeline += [
                URLFilter(
                    extra_domains=robots_domains,
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots",
                        compression="zstd",
                        adapter=finemath_writer_adapter,
                    ),
                    use_integrated_lists=False,
                ),
            ]
        elif "robots_filter" in config and config["robots_filter"] == "fine":
            print("Using fine-grained robots filtering (URL-level)")
            robots_domains = pd.read_parquet(FINE_ROBOTS_PATH)
            robots = {
                row["domain"]: row["content"] for _, row in robots_domains.iterrows()
            }
            del robots_domains
            pipeline += [
                RobotsTxtFilter(
                    robots_dict=robots,
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots",
                        compression="zstd",
                        adapter=finemath_writer_adapter,
                    ),
                ),
            ]

        pipeline += [
            ParquetWriter(
                output_dir,
                compression="zstd",
                adapter=finemath_writer_adapter,
                max_file_size=1 * 2**30,  # 1GB
            ),
        ]

        LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=logs_dir,
            tasks=len(list_files(input_dir)),
            workers=144,
        ).run()
