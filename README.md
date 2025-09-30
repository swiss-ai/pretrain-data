# pretrain-data

This repository contains the pipelines generating the pretraining data used to train Apertus. These pipelines are built on top of [**datatrove**](https://github.com/huggingface/datatrove), a framework for large scale data processing. This project extends **datatrove** with custom modules and Apertus pretraining data pipeline. Note that the pipelines are created to work in the CSCS infrastructure on [**Alps**](https://www.cscs.ch/computers/alps) and needs to be adjusted to specific environment to before run out of the box.

## **Project Structure** 📂

The repository is organized to separate the core code from the examples and documentation.

  - **`src/data_pipeline_pretrain`**: This directory contains all the custom **datatrove** extensions, including new annotators, filters, formatters and tokenizers that are not part of the standard **datatrove** library.
  - **`pipelines/`**: This folder holds the main pipeline definitions. Each subdirectory within `pipelines` corresponds to a specific pretraining data. For details refer to next Section **Pretraining Data**.
  - **`examples/`**: A collection of example scripts demonstrating how to annotate the FineWeb samples with embeddings, toxicity score, code quality or how to tokenize the data to be used in Megatron.

## **How to Run a Pipeline** 🏃‍♂️
> [!IMPORTANT]
> Due to the infrastructure-specific nature of these pipelines (designed for CSCS on Alps), they are not directly runnable out of the box. You will need to adjust the scripts and configurations to your specific computing environment.

### **General Workflow**

1.  **Clone the repository:** First, ensure you have the project cloned to your local environment.
2.  **Set up the environment:** The project's dependencies are defined in the `Dockerfile`. To set up the environment, either build the Docker container as specified or replicate the environment manually based on the `Dockerfile` contents.
3.  **Select a pipeline and review its configuration:** Before running the pipelines, all necessary datasets need to be downloaded first. Then, Navigate to the pipeline you want to run, for example, `pipelines/fineweb-2/main.py`. The paths and configurations are defined within this file. Adjust them according to your environment, including any **SLURM configurations** or by switching to the `LocalPipelineExecutor` for local execution.
4.  **Execute the pipeline:** The pipelines are designed to be executed directly as Python scripts. You can run them in your environment using commands like `python pipelines/euroblocks/main.py` or with specific arguments, such as `python pipelines/fineweb-2/main.py quality_10-filterrobots`.

## **Pretraining Data** ✨
This section lists the specific datasets and the corresponding scripts to process them. If you want to apply `robots.txt` filtering, first download the respective metadata from [swiss-ai/datasets](https://huggingface.co/swiss-ai/datasets).

- **`FineWeb-HQ`**: First Download [FineWeb (v1.3.0)](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/v1.3.0) and the English HQ classifier, then adjust and run `python pipelines/fineweb/main.py quality_33-filterrobots` or `python pipelines/fineweb/main.py quality_33-filterrobots`.
- **`FineWeb2-HQ`**: First Download [FineWeb-2 (v1.0.2)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/v2.0.1) and the multilingual HQ classifiers, then adjust and run `python pipelines/fineweb/main.py quality_33-filterrobots` or `python pipelines/fineweb/main.py quality_33-filterrobots`.
- **`DCLM-Edu`**: 
First Download [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/tree/main/data) and run `python pipelines/dclm-edu/main.py quality_33-filterrobots` or `python pipelines/fineweb/main.py filterrobots_fine`.
- **`FineWeb-Edu`**: First Download [FineWeb-Edu-Score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2/tree/v1.0.0/data) or [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/v1.0.0/data), then adjust and run `python pipelines/fineweb/main.py filterrobots` or `python pipelines/fineweb/main.py filterrobots `.
- **`Code`**: First Download [Starcoder Data](https://huggingface.co/datasets/bigcode/starcoderdata), then run [TBD] 
- **`Math`**: 
First Download [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) or [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath), then adjust and run `python pipelines/finemath/main.py filterrobots_fine` or `python pipelines/megamath/main.py filterrobots`.
- **`Cooldown`** Download [Euroblocks](https://huggingface.co/datasets/utter-project/EuroBlocks-SFT-Synthetic-1124/tree/main) or [Europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl) and run `python pipelines/fineweb/main_bidirectional.py`, `python pipelines/euroblocks/main.py`, respectively. For [Paradocs](https://huggingface.co/datasets/jhu-clsp/paradocs/blob/main/files.yml) you need to adjust and run `python pretrain-data/pipelines/paradocs/preprocessing/run.py`, then `pretrain-data/pipelines/paradocs/main.py`.

Finally to tokenize the data to be used by megatron run:
```bash
python3 examples/preprocess_megatron.py \
--tokenizer-name-or-path alehc/swissai-tokenizer \
--output-folder $output_folder \
--logging-dir $logging_dir \
--n-tasks $number_of_tasks \
--dataset $input_folder \
--column $COLUMN_KEY
```
