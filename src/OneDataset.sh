#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Use: $0 <dataset_name>"
    exit 1
fi

dataset=$1

unzip -o ../data/$dataset/$dataset.zip -d ../data/$dataset

python ClusteringComparison.py -d "$dataset" -m g

python PairwiseComparisons.py -d "$dataset" -m g

python GoldStandardComparison.py -d "$dataset" -m g