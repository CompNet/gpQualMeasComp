#!/bin/bash


datasets=(
    "MUTAG"
    "PTC"
    "AIDS"
    "NCI1"
    "DD"
    "FOPPA"
    "FRANKENSTEIN"
    "IMDB"
)

for dataset in "${datasets[@]}"; do
    # unzip the dataset
    unzip -o ../data/$dataset/$dataset.zip -d ../data/$dataset
    python ClusteringComparison.py -d $dataset -m g
    python PairwiseComparisons.py -d $dataset -m g
    python GoldStandardComparison.py -d $dataset -m g
done
