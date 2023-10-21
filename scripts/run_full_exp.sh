CUDA=$1

TRAINER=$2
DATASET=$3
CFG=""

if [ "$TRAINER" == "CoOp" ]; then
    CFG=vit_b16_ctxv1
    LOADEP=200
elif [ "$TRAINER" == "CoCoOp" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1
    LOADEP=10
elif [ "$TRAINER" == "DoCoOp" ]; then
    #CFG=vit_b16_ctxv1_reduce_maxloss
    #CFG=vit_b16_ctxv1_reduce_minloss
    CFG=vit_b16_ctxv1_reduce_medoids_mmd
    LOADEP=50
elif [ "$TRAINER" == "DoCoCoOp" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1
    LOADEP=10
else
    echo "Invalid TRAINER specified."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 3 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 3 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 3 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 3 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 3 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 4 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 4 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 4 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 4 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 4 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 5 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 5 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 5 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 5 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 5