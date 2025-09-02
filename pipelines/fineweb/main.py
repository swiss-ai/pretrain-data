from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from data_pipeline_pretrain.pipeline.filters import IdFilter, load_robots
from data_pipeline_pretrain.pipeline.filters.embeddings_filter import (
    EmbeddingBinaryClassifierFilter,
    BinaryClassifier,
    estimate_classifier_threshold,
)
from data_pipeline_pretrain.pipeline.filters.toxic_filter import (
    ToxicScorer,
    ToxicityBinaryClassifierFilter,
)
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from data_pipeline_pretrain.utils import list_files, list_folders
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
import sys

# Command-line arguments
if len(sys.argv) != 3:
    print(
        "The first argument should be the config name: python main.py quality_10-keeprobots."
    )
    exit(1)
CONFIG_NAME = sys.argv[1]
DUMP_NR = int(sys.argv[2])

# Filters and configs
QUALITY_CLASSIFIER_PATH = "./models/quality/default/eng_Latn/model.pt"
TOXICITY_CLASSIFIER_PATH = (
    "./multilingual_pretrain/detoxify_models/english_cls.pth"
)
ROBOTS_DIR = "./robotstxt/new_compute_permissivity/fw1_permissivity_logs"

CONFIGS = {
    "quality_10-keeprobots": {
        "robots_filter": False,
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {"threshold": 0.999440610408783},
    },
    "quality_10-filterrobots": {
        "robots_filter": True,
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {"threshold": 0.999440610408783},
    },
    "quality_33-keeprobots": {
        "robots_filter": False,
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {"threshold": 0.999440610408783},
    },
    "quality_33-filterrobots": {
        "robots_filter": True,
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {"threshold": 0.999440610408783},
    },
    "only-quality_10": {
        "quality_filter": {"p": 0.10},
    },
    "only-quality_33": {
        "quality_filter": {"p": 0.33},
    },
    "keeprobots": {
        "robots_filter": False,
    },
    "filterrobots": {
        "robots_filter": True,
    },
}

# Processing directories
EMBEDDINGS_INPUT_DIR = "./embeddings/xlmroberta/fineweb-1_3_0/data/"
RAW_INPUT_DIR = (
    "./fineweb-1_3_0/snapshot/data"
)
BASE_OUTPUT_DIR = f"./swissai-fineweb-1_3_0-{CONFIG_NAME}"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"

# Dumps to process
DUMPS = sorted([folder.name for folder in list_folders(ROBOTS_DIR)], reverse=True)
DUMPS = list(DUMPS)
DUMPS = [DUMPS[DUMP_NR]]

if __name__ == "__main__":
    config = CONFIGS[CONFIG_NAME]

    for dump in DUMPS:
        input_dir = (
            f"{EMBEDDINGS_INPUT_DIR}/{dump}"
            if ("quality_filter" in config or "toxicity_filter" in config)
            else f"{RAW_INPUT_DIR}/{dump}"
        )
        output_dir = f"{OUTPUT_DIR}/output/{dump}"
        removed_dir = f"{OUTPUT_DIR}/removed/{dump}"
        logs_dir = f"{LOGS_DIR}/{dump}/"

        pipeline = [
            ParquetReader(
                input_dir,
            ),
        ]

        if "robots_filter" in config and config["robots_filter"] is True:
            pipeline += [
                IdFilter(
                    ids_to_filter=load_robots(f"{ROBOTS_DIR}/{dump}"),
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots", compression="zstd"
                    ),
                ),
            ]

        if "quality_filter" in config:
            quality_classifier = BinaryClassifier.from_pt(QUALITY_CLASSIFIER_PATH)
            quality_threshold = estimate_classifier_threshold(
                input_dir,
                classifier=quality_classifier,
                num_samples=1_000_000,
                top_p=config["quality_filter"]["p"],
            )
            pipeline += [
                EmbeddingBinaryClassifierFilter(
                    classifier=quality_classifier,
                    threshold=quality_threshold,
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/quality", compression="zstd"
                    ),
                )
            ]

        if "toxicity_filter" in config:
            pipeline += [
                ToxicScorer(model_path=TOXICITY_CLASSIFIER_PATH),
                ToxicityBinaryClassifierFilter(
                    threshold=config["toxicity_filter"]["threshold"],
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/toxicity",
                        compression="zstd",
                    ),
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

        print(f"Launching {dump} with config {config}.")
        print(f"Pipeline: {pipeline}")

        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"fw-{dump}-{CONFIG_NAME}",
            logging_dir=logs_dir,
            tasks=len(list_files(input_dir)),
            time="01:30:00",
            cpus_per_task=6,  # 4 gives OOM, likely due to models & robots
            workers=1,
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
