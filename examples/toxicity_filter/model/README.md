# Toxicity Filtering

## Data Preprocessing

We use the dataset from **[PleIAs](https://arxiv.org/pdf/2410.22587)** and [SWSR](https://zenodo.org/records/4773875) (for Chinese only) and train language-specific classifiers. We subsample non-toxic samples to keep the training set balanced. The preprocessing can be done by running:

```bash
python model/toxicity/preprocessing.py --data_dir <raw_data_dir> --processed_dir <directory to save processed datasets>
```

## Model training

We train a 2-layer MLP model on the binary classification of toxicity. We first get sentence embeddings from `xlm_roberta` model then train the classifier on the top of embeddings. We pulish all the trained checkpoints in [here](https://drive.google.com/drive/folders/1jrNvvbhC5q-1DbLdOcJtpDbTGi-HhJBc?usp=sharing).

You can train your own models by running:

```
python model/toxicity/train_mlp.py --processed_dir <directory to the preprocessed datasets> --model_save_dir <directory to save model checkpoints>
```

## Statistics

We report the test accuracy for each language as follows:

| Language            | Acc.    |
| ------------------- | ------- |
| English (`en`)    | 80.13%  |
| Chinese (`zh`)    | 79.64%  |
| French (`fr`)     | 82.34%  |
| German (`de`)     | 82.61%  |
| Italian (`it`)    | 82.16%  |
| Dutch (`du`)      | 80.94%  |
| Polish (`po`)     | 81.24%  |
| Portuguese (`pt`) | 94.63%  |
| Spanish (`sp`)    | 81.61\% |
