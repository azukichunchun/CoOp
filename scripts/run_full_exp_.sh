CUDA=$1

TRAINER=$2
DATASET=$3
CFG=""

if [ "$TRAINER" == "CoOp" ]; then
    CFG=vit_b16_ctxv1_use_half_class_zhou
    LOADEP=200
elif [ "$TRAINER" == "CoCoOp" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_etran
    LOADEP=10
elif [ "$TRAINER" == "DoCoOp" ]; then
    #CFG=vit_b16_ctxv1_reduce_maxloss
    #CFG=vit_b16_ctxv1_reduce_minloss
    CFG=vit_b16_ctxv1_ep200_reduce_use_full_class_prox_without_dist_loss_zhou
    LOADEP=200
elif [ "$TRAINER" == "DoCoOp2" ]; then
    CFG=vit_b16_ep50_reduce_maxloss_weight_adjust
    LOADEP=200
elif [ "$TRAINER" == "DoCoCoOp" ]; then
    #CFG=vit_b16_ctxv1_ep10_reduce_one_direction_prox_weight_adjust
    #CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_etran_3
    CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_active
    LOADEP=15
elif [ "$TRAINER" == "CLIP-Adater" ]; then
    CFG=vit_b16_c4_ep10_batch1_ctxv1_zhou_active
    LOADEP=200
else
    echo "Invalid TRAINER specified."
    exit 1
fi


# for i in {1..51} ; do
#     echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1 ${i}
#     CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1 ${i}
# done

CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 1 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 1 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 1 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 1
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 2 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 2 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 2 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 2 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 3
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 3
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 3 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 3
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 4 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 4 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 4 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 4 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 4 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 5
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 2 ${CFG} ${LOADEP} 5 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 4 ${CFG} ${LOADEP} 5 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 8 ${CFG} ${LOADEP} 5 &&
# CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 16 ${CFG} ${LOADEP} 5