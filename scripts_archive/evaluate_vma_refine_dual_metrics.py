#!/usr/bin/env python3
"""
双重评测：同时评估VMA和Refined到GT和到点云的距离
1. 到GT距离：评估与标注的一致性
2. 到点云距离：评估与实际数据的拟合质量
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# ==================== 配置 ====================
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_JSON = "/homes/zhangzijian/pointnet_refine/visualizations/dual_metrics.json"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# 评测参数
NUM_SAMPLES_TO_EVAL = 20

# ==================== 工具函数 ====================