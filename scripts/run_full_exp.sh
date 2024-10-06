CUDA=$1

TRAINER=$2
CFG=$3
LOADEP=$4

list=("caltech101" "eurosat" "dtd" "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101" "imagenet")
#list=("imagenet")
for DATASET in ${list[@]}
do

echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1 &&
echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 2 &&
echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 3
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/run_exp.sh ${TRAINER} ${DATASET} 1 ${CFG} ${LOADEP} 3

done