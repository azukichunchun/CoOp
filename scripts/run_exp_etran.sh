#!/bin/bash

# CUDA_VISIBLE_DEVICESの設定
device=$1
dataset=$2

# 実行するメトリックのリスト
metrics=("max" "min" "median" "q1" "q3")

# DoCoCoOpの実験パラメータ
experiment="DoCoCoOp"
version="1"
model="vit_b16_c4_ep10_batch1_ctxv1_zhou_etran"

# 各メトリックに対してループ
for metric in "${metrics[@]}"; do
    for i in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$device bash scripts/run_exp.sh $experiment $dataset $version $model"_"$metric 15 $i
    done
done
