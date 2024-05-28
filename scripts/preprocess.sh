#!/bin/bash

# Add contexts to Stereohoax before split to avoid re-tokenizing the same sentences
python3 context_soft.py

## Stereohoax
data=stereohoax
python3 preprocess_split.py -data $data -datafile train_val_split_context_soft.csv
mv data/$data/clean.csv data/$data/train.csv
python3 preprocess_split.py -data $data -datafile test_split_context_soft.csv
mv data/$data/clean.csv data/$data/test.csv
