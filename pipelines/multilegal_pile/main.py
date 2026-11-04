from pathlib import Path
from datatrove.pipeline.readers import JsonlReader
from data_pipeline_pretrain.pipeline.formatters import PIIFormatter
from datatrove.pipeline.writers import ParquetWriter
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from datatrove.executor import LocalPipelineExecutor


compliant_subsets = """bg/legislation/bulgaria_marcell.jsonl.xz
cs/caselaw/czechia_constitutional_court.jsonl.xz
cs/caselaw/czechia_supreme_administrative_court.jsonl.xz
cs/caselaw/czechia_supreme_court.jsonl.xz
da/caselaw/denmark_ddsc.jsonl.xz
da/legislation/denmark_ddsc.jsonl.xz
de/caselaw/germany_openlegaldata.jsonl.xz
de/caselaw/switzerland_entscheidsuche.jsonl.xz
de/legislation/germany_openlegaldata.jsonl.xz
de/legislation/switzerland_lexfind.jsonl.xz
en/legislation/switzerland_lexfind.jsonl.xz
en/legislation/uk_uk_lex.jsonl.xz
fr/caselaw/belgium_jurportal.jsonl.xz
fr/caselaw/france_cass.jsonl.xz
fr/caselaw/luxembourg_judoc.jsonl.xz
fr/caselaw/switzerland_entscheidsuche.jsonl.xz
fr/legislation/belgium_ejustice.jsonl.xz
fr/legislation/switzerland_lexfind.jsonl.xz
hu/legislation/hungary_marcell.jsonl.xz
it/caselaw/switzerland_entscheidsuche.jsonl.xz
it/legislation/switzerland_lexfind.jsonl.xz
nl/legislation/belgium_ejustice.jsonl.xz
pl/legislation/poland_marcell.jsonl.xz
pt/caselaw/brazil_cjpg_0.jsonl.xz
pt/caselaw/brazil_cjpg_1.jsonl.xz
pt/caselaw/brazil_cjpg_2.jsonl.xz
pt/caselaw/brazil_cjpg_3.jsonl.xz
pt/caselaw/brazil_cjpg_4.jsonl.xz
pt/caselaw/brazil_creta.jsonl.xz
pt/caselaw/brazil_rulingbr.jsonl.xz
ro/legislation/romania_marcell.jsonl.xz
sk/legislation/slovakia_marcell.jsonl.xz
sl/legislation/slovenia_marcell.jsonl.xz"""

DEBUG = False

if __name__ == "__main__":
    # Specify these directories as needed
    DATA_DIR = Path("./joelniklaus/Multi_Legal_Pile/data")
    OUTPUT_DIR = Path("./Multi_Legal_Pile-preprocessed/data")

    # Write the compliant subsets to a file for the JsonlReader
    compliant_subsets_file = OUTPUT_DIR.parent / "compliant_subsets.txt"
    compliant_subsets_file.parent.mkdir(parents=True, exist_ok=True)
    compliant_subsets_file.write_text(compliant_subsets)

    REMOVED_DIR = Path("./Multi_Legal_Pile-preprocessed/removed")


    # Build the processing pipeline
    # Read the files
    def custom_adapter(self, data: dict, path: str, id_in_file: int | str):
        import json
        return {
            "text": data.pop(self.text_key, ""),
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": json.loads(data.pop("metadata", "{}"))
                        | data,  # remaining data goes into metadata
        }


    pipeline = [JsonlReader(data_folder="./Multi_Legal_Pile/data",
                            paths_file=str(compliant_subsets_file),
                            adapter=custom_adapter)]

    # PII formatting
    pipeline += [PIIFormatter()]
    pipeline += [ParquetWriter(str(OUTPUT_DIR),
                               compression="zstd",
                               max_file_size=1 * 2 ** 30)]

    if DEBUG:

        LocalPipelineExecutor(pipeline=pipeline).run()

    else:
        SlurmPipelineNodeExecutor(
            pipeline=pipeline,
            job_name="Multi_Legal_Pile_pipeline",
            logging_dir=f"{OUTPUT_DIR.parent / 'logs'}",
            tasks=len(compliant_subsets_file.read_text().splitlines()),
            time="03:00:00",  # Adjust time as needed
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