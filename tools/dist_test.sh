#!/usr/bin/env bash
cd path-to-project
CONFIG=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
CHECKPOINT=work_dirs/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
work_dir=work_dirs/mm_grounding_dino
GPUS=8

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
PORT=${MASTER_PORT}
MASTER_ADDR=${MASTER_ADDR}

mkdir $work_dir
log_file=${work_dir}/test_log_rank${RANK}.txt
echo "Logging from rank $node_rank" > $log_file 2>&1 &

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --work-dir $work_dir\
    --launcher pytorch ${@:3} > $log_file 