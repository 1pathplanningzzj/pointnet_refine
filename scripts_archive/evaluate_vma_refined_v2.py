#!/usr/bin/env python3
"""
Evaluate VMA and Refined results against GT using chamfer distance metric.
Adapted to work with the refined results format from refine_vma_results.py
"""
import json
import numpy as np
from scipy.spatial import distance
from shapely.geometry import LineString
import os

def custom_polyline_score(pred_lines, gt_lines, metric='chamfer'):
    """
    Calculate similarity between predicted segments and GT lines.

    Args:
        pred_lines: (num_preds, npts, 2) array - short segments
        gt_lines: (num_gts, npts, 2) array - long global lines
        metric: 'chamfer' distance metric

    Returns:
        iou_matrix: (num_preds, num_gts) similarity matrix
