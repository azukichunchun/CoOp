CUDA=$1
TRAINER=$2
DATASET=$3

if [ "$TRAINER" == "CoOp" ]; then
    CFG=vit_b16_ctxv1_use_full_class_zhou_xd    
    LOADEP=200
elif [ "$TRAINER" == "CoCoOp" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_2_xd
    LOADEP=10
elif [ "$TRAINER" == "DoCoCoOp" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_xd
    LOADEP=15
else
    echo "Invalid TRAINER specified."
    exit 1
fi


CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/xd_train.sh ${TRAINER} ${DATASET} ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/xd_train.sh ${TRAINER} ${DATASET} ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/xd_train.sh ${TRAINER} ${DATASET} ${CFG} ${LOADEP} 3