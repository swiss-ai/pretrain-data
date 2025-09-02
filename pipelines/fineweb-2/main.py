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
from datatrove.pipeline.filters import SamplerFilter
from datatrove.executor.local import LocalPipelineExecutor
import sys

# Command-line arguments
if len(sys.argv) != 2:
    print(
        "The first argument should be the config name: python main.py quality_10-keeprobots."
    )
    exit(1)
CONFIG_NAME = sys.argv[1]

# Filters and configs
QUALITY_CLASSIFIER_DIR = (
    "./models/swissai-quality/default/"
)
TOXICITY_CLASSIFIER_DIR = (
    "./multilingual_pretrain/detoxify_models/"
)
ROBOTS_DIR = "./new_compute_permissivity/fw2_permissivity_logs"
CONFIGS_10 = {
    "deu_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9977,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/german_cls.pth",
        },
    },
    "fra_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9994,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/french_cls.pth",
        },
    },
    "pol_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9867,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/polish_cls.pth",
        },
    },
    "por_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9476,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/portuguese_cls.pth",
        },
    },
    "spa_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9953,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/spanish_cls.pth",
        },
    },
    "ita_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9985,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/italian_cls.pth",
        },
    },
    "cmn_Hani": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.5954,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/chinese_cls.pth",
        },
    },
    "nld_Latn": {
        "quality_filter": {"p": 0.1},
        "toxicity_filter": {
            "threshold": 0.9883,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/dutch_cls.pth",
        },
    },
    "rus_Cyrl": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "jpn_Jpan": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "ind_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "tur_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "ces_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "vie_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "swe_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "fas_Arab": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "arb_Arab": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "ell_Grek": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "dan_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "hun_Latn": {"quality_filter": {"p": 0.1}, "sampler": {"rate": 0.95}},
    "_default": {"sampler": {"rate": 0.095}},
}
CONFIGS_33 = {
    "deu_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9977,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/german_cls.pth",
        },
    },
    "fra_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9994,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/french_cls.pth",
        },
    },
    "pol_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9867,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/polish_cls.pth",
        },
    },
    "por_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9476,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/portuguese_cls.pth",
        },
    },
    "spa_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9953,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/spanish_cls.pth",
        },
    },
    "ita_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9985,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/italian_cls.pth",
        },
    },
    "cmn_Hani": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.5954,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/chinese_cls.pth",
        },
    },
    "nld_Latn": {
        "quality_filter": {"p": 0.33},
        "toxicity_filter": {
            "threshold": 0.9883,
            "model_path": f"{TOXICITY_CLASSIFIER_DIR}/dutch_cls.pth",
        },
    },
    "rus_Cyrl": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "jpn_Jpan": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "ind_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "tur_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "ces_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "vie_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "swe_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "fas_Arab": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "arb_Arab": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "ell_Grek": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "dan_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "hun_Latn": {"quality_filter": {"p": 0.33}, "sampler": {"rate": 0.95}},
    "_default": {"sampler": {"rate": 0.3135}},
}
CONFIGS = {
    "quality_10-keeprobots": {
        k: v | {"robots_filter": False} for k, v in CONFIGS_10.items()
    },
    "quality_10-filterrobots": {
        k: v | {"robots_filter": True} for k, v in CONFIGS_10.items()
    },
    "quality_33-keeprobots": {
        k: v | {"robots_filter": False} for k, v in CONFIGS_33.items()
    },
    "quality_33-filterrobots": {
        k: v | {"robots_filter": True} for k, v in CONFIGS_33.items()
    },
    "keeprobots": {
        "_default": {
            "robots_filter": False,
        },
    },
    "filterrobots": {
        "_default": {
            "robots_filter": True,
        },
    },
}

# Processing directories
EMBEDDINGS_INPUT_DIR = "./mfineweb-draft/embeddings/xlmroberta/fineweb-2_0_1/"
RAW_INPUT_DIR = (
    "./fineweb-2_0_1/snapshot/data/"
)
BASE_OUTPUT_DIR = f"./swissai-fineweb-2-{CONFIG_NAME}"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/data/"
LOGS_DIR = f"{BASE_OUTPUT_DIR}/logs/"

# Languages to process
LANGUAGES = [
    folder.name
    for folder in list_folders(RAW_INPUT_DIR)
    if not folder.name.endswith("_removed")
]

if __name__ == "__main__":
    last_executor = None

    for language in LANGUAGES:
        config = (
            CONFIGS[CONFIG_NAME][language]
            if language in CONFIGS[CONFIG_NAME]
            else CONFIGS[CONFIG_NAME]["_default"]
        )

        input_dir = (
            f"{EMBEDDINGS_INPUT_DIR}/{language}"
            if ("quality_filter" in config or "toxicity_filter" in config)
            else f"{RAW_INPUT_DIR}/{language}/train"
        )
        output_dir = f"{OUTPUT_DIR}/output/{language}"
        removed_dir = f"{OUTPUT_DIR}/removed/{language}"
        logs_dir = f"{LOGS_DIR}/{language}/"

        pipeline = [
            ParquetReader(
                input_dir,
            ),
        ]

        if "robots_filter" in config and config["robots_filter"] is True:
            pipeline += [
                IdFilter(
                    ids_to_filter=load_robots(f"{ROBOTS_DIR}/{language}/train"),
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/robots", compression="zstd"
                    ),
                ),
            ]

        if "quality_filter" in config:
            quality_classifier = BinaryClassifier.from_pt(
                f"{QUALITY_CLASSIFIER_DIR}/{language}/model.pt"
            )
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
                ToxicScorer(model_path=config["toxicity_filter"]["model_path"]),
                ToxicityBinaryClassifierFilter(
                    threshold=config["toxicity_filter"]["threshold"],
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/toxicity",
                        compression="zstd",
                    ),
                ),
            ]

        if "sampler" in config:
            pipeline += [
                SamplerFilter(
                    rate=config["sampler"]["rate"],
                    seed=42,
                    exclusion_writer=ParquetWriter(
                        f"{removed_dir}/sampler", compression="zstd"
                    ),
                )
            ]

        pipeline += [
            PIIFormatter(),
            ParquetWriter(
                output_dir,
                compression="zstd",
                max_file_size=1 * 2**30,  # 1GB
            ),
        ]

        print(f"Launching {language} with config {config}.")
        print(pipeline)

        # Launches one job per language with dependencies (one runs at a time) => lots of jobs
        last_executor = SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name=f"fw2-{language}-{CONFIG_NAME}",
            logging_dir=logs_dir,
            tasks=len(list_files(input_dir)),
            time="08:00:00",
            cpus_per_task=4,
            randomize_start_duration=15,
            partition="normal",
            srun_args={
                "environment": "",
                "account": "",
            },
            run_on_dependency_fail=True,
            depends=last_executor,
        )
        last_executor.run()
