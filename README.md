# pretrain-data

Scripts used to build the [Apertus](https://huggingface.co/swiss-ai) pretraining corpus with [datatrove](https://github.com/huggingface/datatrove).

## Repository layout

- `pipelines/`: dataset-specific processing scripts for FineWeb, FineWeb-2, FineWeb-Edu, DCLM-Edu, FineMath, MegaMath, Europarl, ParaDocs, and other sources.
- `src/data_pipeline_pretrain/`: reusable datatrove components for SLURM execution, filtering, PII formatting, embedding annotation, and Megatron tokenization.
- `examples/`: supporting workflows for embeddings, toxicity and code filtering, and tokenization.

## Running a pipeline

These scripts were written for the CSCS Alps cluster. Paths and SLURM accounts were replaced with placeholders for publication, so configure them for your environment before running. Source datasets, robots.txt metadata, embeddings, and model checkpoints must be provided separately where needed.

1. Install the dependencies imported by the chosen script and add the local library to `PYTHONPATH`:

   ```bash
   export PYTHONPATH="$PWD/src:$PYTHONPATH"
   ```

2. Set the input, output, model, and SLURM paths in the script.
3. Choose a config from its `CONFIGS` dictionary and run it. For example:

   ```bash
   python pipelines/fineweb-2/main.py quality_33-filterrobots
   ```

The scripts are a reference for the original processing workflows; they require adaptation to run elsewhere.

## License

Apache 2.0 — see [LICENSE](LICENSE).
