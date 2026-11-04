from pathlib import Path
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor


if __name__ == "__main__":
    # Specify these directories as needed
    DATA_DIR = Path("./nvidia/Nemotron-CC-v2.1")
    OUTPUT_DIR = Path("./Nemotron-CC-v2.1-preprocessed/data")

    # Write the compliant subsets to a file for the JsonlReader
    subsets = ['High-Quality', 'High-Quality-DQA']
    subsets_file = OUTPUT_DIR.parent / "files_path.txt"
    subsets_paths = [p.relative_to(DATA_DIR)
                     for subset in subsets
                     for p in (DATA_DIR / subset).rglob("*.parquet")]

    subsets_file.parent.mkdir(parents=True, exist_ok=True)
    subsets_file.write_text("\n".join(str(p) for p in subsets_paths))


    pipeline = [ParquetReader(str(DATA_DIR), paths_file=str(subsets_file))]

    # PII formatting
    pipeline += [PIIFormatter()]
    pipeline += [ParquetWriter(str(OUTPUT_DIR),
                               compression="zstd",
                               max_file_size=1* 2**30)]

    SlurmPipelineNodeExecutor(
        pipeline=pipeline,
        job_name="Nemotron-CC-v2.1_pipeline",
        logging_dir=f"{OUTPUT_DIR.parent / 'logs'}",
        tasks=len(subsets_paths),
        time="06:00:00",  # Adjust time as needed
        cpus_per_task=6,
        randomize_start_duration=15,
        workers=8,
        partition="normal",
        srun_args={
            "environment": "",
            "account": "",
        },
        sbatch_args={
            "account": "",
        },
    ).run()