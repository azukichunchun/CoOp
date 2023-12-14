#!/bin/bash

# custom config
DATA=data

TRAINER=$1
DATASET=$2
SHOTS=$3
CFG=$4
LOADEP=$5
SEED=$6

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_new/${COMMON_DIR}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--embedding_feature \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all