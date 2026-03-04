#!/usr/bin/env bash
cd path-to-project

work_dir=work_dirs/weight-distill-20
CONFIG=configs/detr_vip/DETR-ViP_swin-t_pretrain_obj365_goldg.py
GPUS=4

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
PORT=${MASTER_PORT}
MASTER_ADDR=${MASTER_ADDR}

mkdir $work_dir
log_file=${work_dir}/train_log_rank${RANK}.txt
echo "Logging from rank $node_rank" > $log_file 2>&1 &

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --work-dir $work_dir\
    --launcher pytorch ${@:3} > $log_file 