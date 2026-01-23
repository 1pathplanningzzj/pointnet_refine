#!/bin/bash

PY_BIN=/root/miniconda3/envs/pointnet_refine/bin/python

# Fixed checkpoint directory
CHECKPOINT_DIR="/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based"

unset PYTHONPATH
export PYTHONPATH=/homes/zhangzijian/pointnet_refine:$PYTHONPATH

cd /homes/zhangzijian/pointnet_refine

echo "Using Python: $PY_BIN"
echo "Saving checkpoints to: $CHECKPOINT_DIR"

$PY_BIN -m torch.distributed.run --nproc_per_node=8 train_dist.py --checkpoints_dir "$CHECKPOINT_DIR"
