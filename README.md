# Human vs. Machine Perceptions on Immigration Stereotypes [Link to paper](https://aclanthology.org/2024.lrec-main.741/)

This repository contains the code for classification models of racial stereotypes in a Spanish corpus (StereoHoax-ES), using both hard and soft labels.

**Disclaimer**: the original repository was used with more corpora and options than the ones presented on the article above.
For example:

- Task 1 refers to stereotype prediction (yes/no) [**this paper**].
- Task 2 was the stereotype impliciteness classification (imp/exp), see [DETESTS-Dis: DETEction and classification of racial STereotypes in Spanish - Learning with Disagreement](https://detests-dis.github.io/).

## Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Programs and folder structure](#programs-and-folder-structure)
  - [Main programs](#main-programs)
  - [Utils](#utils)
- [StereoHoax-ES corpus](#stereohoax-es-corpus)
- [Results](#results)
- [Reproduce](#reproduce)
  - [Setup](#setup)
  - [Split](#split)
  - [Baselines](#baselines)
  - [BERTs Fine-tuning](#berts-fine-tuning)
    - [Hyper-parameters](#hyper-parameters)
  - [Figures](#figures)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Programs and folder structure


### Main programs

- `baselines.py`: Creates the [baselines](#baselines) for the tasks.
- `context_soft.py` Adds the context attributes and soft-labels to the datasets.
- `create_figures.py` Create figues, stored in `results/figures`.
- `create_metrics.py`: Computes the metrics for the results files.
- `preprocess_split.py`: Preprocesses and splits the corpus as needed.

### Utils

- `config.py`: Contains feature names and constants.
- `data_pred.py`: Does the data processing and predictions for the fine-tuned models.
- `fine_tuning.py` Utils for the fine-tuning notebook.
- `io.py`: Parses inputs and outputs.
- `plots.py`: Plot utils.
- `results.py`: Contains a class to store the results and functions to compute
  metrics.
- `split.py`: Utils for split notebooks.
- `trainers.py`: Different HF Trainers and main train function.

- `scripts`: This folder contains bash scripts for [reproducibility](#reproduce).


The following notebooks are included:

- `notebooks/split_stereohoax.py` splits the StereoHoax corpus (see
  [split](#split)).
- [`notebooks/fine_tuning_hard_and_soft.py`](https://colab.research.google.com/drive/1vKW90aaYxsjUS-njrR0s8ciyZ590D3nK?usp=sharing)
  has the fine-tuning of the BERT models. It was originally run on free Google Colab with a T4 GPU.
  It was since ported to be used on a local machine.

## StereoHoax-ES corpus

If you are interested in the corpus **for research purposes** please [contact the authors](mailto:pol.pastells@ub.edu),
and we will provide the zip password.

- `stereoHoax-ES_goldstandard.csv` is the original data.
- `stereohoax_unaggregated.csv` has the 3 unaggregated annotations for
  "stereotype","implicit", "contextual" and "url".
- `train_val_split.csv`, `train_split.csv`, `val_split.csv` and `test_split.csv`
  are the [split](#split) sets, also with the unaggregated annotations.
  1. `context_soft.py` creates a `_context_soft` version for each one.
  2. `preprocess_split` takes `train_val_split_context_soft.csv` and
     `test_split_context_soft.csv` as inputs to create `train.csv` and
     `test.csv`.

## Results

Predictions for each model are stored in the `results` folder along with the
gold standard. Each model's results are separated into different CSV files,
with the predictions for each feature being named '<feature>\_pred'.

The overall metrics for all models are shown in the `results/metrics` folder.

These metrics can be recreated using the `create_metrics.py` script.

## Reproduce

### Setup

To set up the necessary environment and download required models, run
`scripts/setup.sh`.

### Split

For the baselines, the data is simply split into train and test sets. For the
BERT models, a validation set is also created.

For the StereoHoax corpus, the following splits are created: 70% train, 10%
validation, and 20% test. Different racial hoaxes are separated into different
sets to avoid data leakage and preserve the distribution of stereotypes. The
split is performed using `split_stereohoax.py`.

`split_stereohoax.py` works in the following way:

1. Finds combination of hoaxes that reach 70%, 20% and 10% of the data.
2. Finds which of these combinations has the most similar topic distribution to
   the original data.

The resulting splits are the following:

- Train_val - 80% = Train + val (used for baselines)
- Train - 70%: 'SP003', 'SP013', 'SP064', 'SP054', 'SP070', 'SP017', 'SP067',
  'SP043', 'SP036', 'SP048'
- Val - 10%: 'SP005', 'SP065', 'SP052', 'SP055', 'SP068'
- Test - 20%: 'SP057', 'SP015', 'SP049', 'SP047', 'SP010', 'SP014', 'SP009',
  'SP027', 'SP040', 'SP020', 'SP023', 'SP008', 'SP031'

The percentage of the whole dataset that each racial hoax (RH) contributes to
the splits is the following:

```python
Train = {
    'SP067': 0.93,
    'SP043': 27.97,
    'SP036': 8.32,
    'SP048': 0.06,
    'SP064': 14.73,
    'SP003': 16.69,
    'SP054': 0.02,
    'SP070': 0.04,
    'SP013': 0.02,
    'SP017': 0.07,
    'sum':  68.85,
}
Validation = {
    'SP052': 1.72,
    'SP068': 3.87,
    'SP005': 0.06,
    'SP065': 2.13,
    'SP055': 3.33,
    'sum':  11.10,
}
Test = {
    'SP010': 0.19,
    'SP008': 1.31,
    'SP014': 1.42,
    'SP027': 5.12,
    'SP015': 3.72,
    'SP009': 0.50,
    'SP040': 0.79,
    'SP031': 0.24,
    'SP020': 1.70,
    'SP023': 0.34,
    'SP047': 3.44,
    'SP049': 1.03,
    'SP057': 0.24,
    'sum':  20.04,
}
```

### Baselines

`train.csv` and `test.csv` are used for the baselines. To obtain them run
`scripts/preprocess.sh`, which calls `preprocess_split.py`.

The following baselines are considered:

- All-zeros
- All-ones
- Weighted random classifier
- TFIDF (with only unigrams) + linear SVC (Support Vector Classifier)
- TFIDF (with n-grams with sizes 1 to 3) + linear SVC
- FastText vectorization + linear SVC

They can be run with: `python baselines.py`


### BERTs Fine-tuning

To fine-tune the BERT models for both corpora, run the `fine_tuning_hard_and_soft.ipynb` notebook with the adequate inputs.

The BERT models used for these tasks include:

- 'PlanTL-GOB-ES/roberta-base-bne' (RoBERTa-BNE). Trained with data from the
  [National Library of Spain (Biblioteca Nacional de España)](https://www.bne.es/es).
- 'dccuchile/bert-base-spanish-wwm-cased' (BETO). Trained with the data
  specified in https://github.com/josecannete/spanish-corpora.
- 'pysentimiento/robertuito-base-uncased' (RoBERTuito). Tained with
  [twitter data](https://arxiv.org/pdf/2111.09453.pdf).

#### Hyper-parameters

The models with hard labels uses a learning rate of 2e-5, while the model with
soft-labels uses 1e-5.

We keep the model with lowest loss.

### Figures

To create all the figures run:

```bash
python create_metrics.py
python create_figures.py
```
