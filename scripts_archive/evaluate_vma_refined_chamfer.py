#!/usr/bin/env python3
"""
Evaluate VMA and Refined results against GT using chamfer distance metric.
Following the evaluation method from vma-dev.
"""
import json
import numpy as np
from scipy.spatial import distance
from shapely.geometry import LineString
from shapely.strtree import STRtree
from shapely.geometry import CAP_STYLE, JOIN_STYLE
import matplotlib.pyplot as plt

def custom_polyline_score(pred_lines, gt_lines, linewidth=2., metric='chamfer'):
    """
    Calculate similarity between predicted segments and GT lines.

    For segmented predictions vs global GT:
    - Extract the GT portion that overlaps with pred segment's X range
    - Compute chamfer distance only on the overlapping portion

    Args:
        pred_lines: (num_preds, npts, 2) array - short segments
        gt_lines: (num_gts, npts, 2) array - long global lines
        linewidth: not used
        metric: 'chamfer' distance metric

    Returns:
        iou_matrix: (num_preds, num_gts) similarity matrix
                   For chamfer: negative average distance (closer to 0 is better)
    """
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    if metric == 'chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    else:
        raise NotImplementedError

    # Compute chamfer distance for all pairs
    for i in range(num_gts):
        gt_line = gt_lines[i]
        gt_x_min, gt_x_max = gt_line[:, 0].min(), gt_line[:, 0].max()
        gt_y_min, gt_y_max = gt_line[:, 1].min(), gt_line[:, 1].max()

        for j in range(num_preds):
            pred_line = pred_lines[j]
            pred_x_min, pred_x_max = pred_line[:, 0].min(), pred_line[:, 0].max()
            pred_y_min, pred_y_max = pred_line[:, 1].min(), pred_line[:, 1].max()

            # Check if there's spatial overlap
            x_overlap = (pred_x_min <= gt_x_max) and (pred_x_max >= gt_x_min)
            y_overlap = (pred_y_min <= gt_y_max) and (pred_y_max >= gt_y_min)

            if not (x_overlap and y_overlap):
                continue

            if metric == 'chamfer':
                # Extract GT points within pred's X range (with some margin)
                margin = 10.0  # 10m margin
                x_min_range = pred_x_min - margin
                x_max_range = pred_x_max + margin

                # Filter GT points within X range
                gt_mask = (gt_line[:, 0] >= x_min_range) & (gt_line[:, 0] <= x_max_range)
                gt_filtered = gt_line[gt_mask]

                if len(gt_filtered) < 2:
                    continue

                # Compute pairwise distances
                dist_mat = distance.cdist(pred_line, gt_filtered, 'euclidean')

                # Bidirectional chamfer distance
                valid_ab = dist_mat.min(-1).mean()  # pred to gt
                valid_ba = dist_mat.min(-2).mean()  # gt to pred

                iou_matrix[j, i] = -(valid_ab + valid_ba) / 2

    return iou_matrix


def custom_tpfp_gen(gen_lines, gt_lines, threshold=0.5, metric='chamfer'):
    """
    Check if detected lines are true positive or false positive.

    Args:
        gen_lines: (num_gens, npts*2+1) array, last column is confidence score
        gt_lines: (num_gts, npts*2) array
        threshold: matching threshold (for chamfer, will be converted to negative)
        metric: 'chamfer' distance metric

    Returns:
        tp: (num_gens,) binary array indicating true positives
        fp: (num_gens,) binary array indicating false positives
        tp_gt: (num_gts,) indices of matched predictions
        gt_covered: (num_gts,) boolean array indicating which GTs are matched
    """
    if metric == 'chamfer':
        if threshold > 0:
            threshold = -threshold  # Convert to negative for chamfer distance

    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # Initialize arrays
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)
    tp_gt = np.zeros((num_gts), dtype=np.int32)
    gt_covered = np.zeros(num_gts, dtype=bool)

    if num_gts == 0:
        fp[...] = 1
        return tp, fp, tp_gt, gt_covered

    if num_gens == 0:
        return tp, fp, tp_gt, gt_covered

    gen_scores = gen_lines[:, -1]  # Extract confidence scores

    # Compute similarity matrix
    matrix = custom_polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, 2),
        gt_lines.reshape(num_gts, -1, 2),
        linewidth=2.,
        metric=metric)

    # For each detection, find the best matching GT
    matrix_max = matrix.max(axis=1)
    matrix_argmax = matrix.argmax(axis=1)

    # Sort detections by confidence score (descending)
    sort_inds = np.argsort(-gen_scores)

    # Match detections to GTs
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                tp_gt[matched_gt] = i
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp, tp_gt, gt_covered


def load_gt_lines_from_annots(vma_file, annot_dir, num_sample=100):
    """Load per-frame GT annotations from annot files (segmented GT matching VMA frames)."""
    with open(vma_file, 'r') as f:
        vma_data = json.load(f)

    gt_lines = []
    gt_timestamps = []

    for timestamp in vma_data.keys():
        timestamp_key = timestamp.split('/')[-1].replace('.jpg', '')
        annot_file = f"{annot_dir}/{timestamp_key}.json"

        try:
            with open(annot_file, 'r') as f:
                annot = json.load(f)

            # Extract GT lines from instances
            if 'instances' in annot:
                for inst in annot['instances']:
                    if 'semantic_line' in inst and inst['semantic_line'] is not None:
                        if 'position' in inst['semantic_line']:
                            points = []
                            for pt in inst['semantic_line']['position']:
                                points.append([pt['x'], pt['y']])
                            if len(points) >= 2:
                                # Resample to fixed number of points
                                line = LineString(points)
                                distances = np.linspace(0, line.length, num_sample)
                                sampled_points = np.array([
                                    list(line.interpolate(distance).coords)
                                    for distance in distances
                                ]).reshape(-1, 2)
                                gt_lines.append(sampled_points)
                                gt_timestamps.append(timestamp)
        except FileNotFoundError:
            continue

    return gt_lines, gt_timestamps


def load_gt_lines(gt_file, num_sample=100):
    """Load GT annotations and resample to fixed number of points."""
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    gt_lines = []
    for item in gt_data['items']:
        if 'semantic_line' in item and item['semantic_line'] is not None:
            if 'position' in item['semantic_line']:
                points = []
                for pt in item['semantic_line']['position']:
                    points.append([pt['x'], pt['y']])
                if len(points) >= 2:
                    # Resample to fixed number of points
                    line = LineString(points)
                    distances = np.linspace(0, line.length, num_sample)
                    sampled_points = np.array([
                        list(line.interpolate(distance).coords)
                        for distance in distances
                    ]).reshape(-1, 2)
                    gt_lines.append(sampled_points)

    return gt_lines


def load_vma_lines(vma_file, annot_dir, num_sample=100):
    """Load VMA detection results and transform to world coordinates."""
    with open(vma_file, 'r') as f:
        vma_data = json.load(f)

    # Load transformation matrices
    first_timestamp = list(vma_data.keys())[0].split('/')[-1].replace('.jpg', '')
    annot_file = f"{annot_dir}/{first_timestamp}.json"

    with open(annot_file, 'r') as f:
        annot = json.load(f)

    M_cropped_reverse = np.array(annot['M_cropped_reverse'])
    M_bev2world = np.array(annot['M_bev2world'])

    def transform_line_coord_2d(line, matrix):
        line = np.array(line)
        x, y = line[:, 0], line[:, 1]
        pts = np.vstack([x, y, np.ones_like(x)])
        pts_trans = matrix @ pts
        return pts_trans[:2].T

    def image_to_world(points_2d, M_cropped_reverse, M_bev2world):
        points_bev = transform_line_coord_2d(points_2d, M_cropped_reverse)
        points_world = transform_line_coord_2d(points_bev, M_bev2world)
        return points_world

    vma_lines = []
    vma_scores = []
    for timestamp, data in vma_data.items():
        pred_instances = data['pred_instances']
        for line_data in pred_instances:
            points_2d = np.array(line_data['data'])
            points_world = image_to_world(points_2d, M_cropped_reverse, M_bev2world)

            # Resample to fixed number of points
            if len(points_world) >= 2:
                line = LineString(points_world)
                distances = np.linspace(0, line.length, num_sample)
                sampled_points = np.array([
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]).reshape(-1, 2)
                vma_lines.append(sampled_points)
                vma_scores.append(line_data.get('confidence_level', 1.0))

    return vma_lines, vma_scores


def load_refined_lines(refined_file, num_sample=100):
    """Load refined results (already in world coordinates)."""
    with open(refined_file, 'r') as f:
        refined_data = json.load(f)

    refined_lines = []
    refined_scores = []
    for item in refined_data:
        points = np.array(item['refined_points'])
        if len(points) >= 2:
            # Resample to fixed number of points
            line = LineString(points)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array([
                list(line.interpolate(distance).coords)
                for distance in distances
            ]).reshape(-1, 2)
            refined_lines.append(sampled_points)
            refined_scores.append(item.get('confidence_level', 1.0))

    return refined_lines, refined_scores


def evaluate_lines(pred_lines, pred_scores, gt_lines, threshold=0.5, metric='chamfer'):
    """
    Evaluate predicted lines against GT.

    Returns:
        dict with precision, recall, AP, and other metrics
    """
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    if num_preds == 0 or num_gts == 0:
        return {
            'num_preds': num_preds,
            'num_gts': num_gts,
            'precision': 0.0,
            'recall': 0.0,
            'ap': 0.0,
            'f1': 0.0
        }

    # Convert to format expected by tpfp function
    pred_lines_array = np.array(pred_lines)
    pred_scores_array = np.array(pred_scores)[:, np.newaxis]
    gen_lines = np.concatenate([
        pred_lines_array.reshape(num_preds, -1),
        pred_scores_array
    ], axis=-1)

    gt_lines_array = np.array(gt_lines).reshape(num_gts, -1)

    # Compute TP/FP
    tp, fp, tp_gt, gt_covered = custom_tpfp_gen(
        gen_lines, gt_lines_array, threshold=threshold, metric=metric)

    # Sort by confidence score
    sort_inds = np.argsort(-pred_scores_array[:, 0])
    tp_sorted = tp[sort_inds]
    fp_sorted = fp[sort_inds]

    # Cumulative TP/FP
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(fp_sorted)

    # Precision and recall
    eps = np.finfo(np.float32).eps
    recalls = tp_cumsum / np.maximum(num_gts, eps)
    precisions = tp_cumsum / np.maximum((tp_cumsum + fp_cumsum), eps)

    # Calculate AP using area under PR curve
    recalls = np.concatenate(([0.], recalls, [recalls[-1]]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate area
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    # Final precision and recall
    final_precision = precisions[-2] if len(precisions) > 1 else 0.0
    final_recall = recalls[-2] if len(recalls) > 1 else 0.0
    f1 = 2 * final_precision * final_recall / (final_precision + final_recall + eps)

    return {
        'num_preds': num_preds,
        'num_gts': num_gts,
        'precision': final_precision,
        'recall': final_recall,
        'ap': ap,
        'f1': f1,
        'tp_count': int(tp.sum()),
        'fp_count': int(fp.sum())
    }


def main():
    # File paths
    gt_file = 'data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json'
    vma_file = 'data/vma_test_data/results_fuse175_test316.json'
    refined_file = 'results_refined_epoch35.json'
    annot_dir = 'data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/cropped_data/annots'

    num_sample = 100  # Resample all lines to 100 points
    threshold = 0.5  # 0.5 meter threshold
    metric = 'chamfer'

    print("Loading data...")

    # Load segmented GT from annot files (per-frame GT)
    gt_lines, gt_timestamps = load_gt_lines_from_annots(vma_file, annot_dir, num_sample)
    print(f"Loaded {len(gt_lines)} GT lines from per-frame annotations")

    vma_lines, vma_scores = load_vma_lines(vma_file, annot_dir, num_sample)
    print(f"Loaded {len(vma_lines)} VMA lines")

    try:
        refined_lines, refined_scores = load_refined_lines(refined_file, num_sample)
        print(f"Loaded {len(refined_lines)} Refined lines")
    except FileNotFoundError:
        print(f"Refined file not found: {refined_file}")
        refined_lines, refined_scores = [], []

    # Evaluate VMA
    print(f"\n{'='*60}")
    print(f"Evaluating VMA results (threshold={threshold}m, metric={metric})")
    print(f"{'='*60}")
    vma_results = evaluate_lines(vma_lines, vma_scores, gt_lines, threshold, metric)
    print(f"VMA Results:")
    print(f"  Predictions: {vma_results['num_preds']}")
    print(f"  Ground Truth: {vma_results['num_gts']}")
    print(f"  True Positives: {vma_results['tp_count']}")
    print(f"  False Positives: {vma_results['fp_count']}")
    print(f"  Precision: {vma_results['precision']:.4f}")
    print(f"  Recall: {vma_results['recall']:.4f}")
    print(f"  F1-Score: {vma_results['f1']:.4f}")
    print(f"  AP: {vma_results['ap']:.4f}")

    # Evaluate Refined
    if refined_lines:
        print(f"\n{'='*60}")
        print(f"Evaluating Refined results (threshold={threshold}m, metric={metric})")
        print(f"{'='*60}")
        refined_results = evaluate_lines(refined_lines, refined_scores, gt_lines, threshold, metric)
        print(f"Refined Results:")
        print(f"  Predictions: {refined_results['num_preds']}")
        print(f"  Ground Truth: {refined_results['num_gts']}")
        print(f"  True Positives: {refined_results['tp_count']}")
        print(f"  False Positives: {refined_results['fp_count']}")
        print(f"  Precision: {refined_results['precision']:.4f}")
        print(f"  Recall: {refined_results['recall']:.4f}")
        print(f"  F1-Score: {refined_results['f1']:.4f}")
        print(f"  AP: {refined_results['ap']:.4f}")

        # Compare
        print(f"\n{'='*60}")
        print(f"Improvement (Refined vs VMA)")
        print(f"{'='*60}")
        print(f"  Precision: {refined_results['precision'] - vma_results['precision']:+.4f}")
        print(f"  Recall: {refined_results['recall'] - vma_results['recall']:+.4f}")
        print(f"  F1-Score: {refined_results['f1'] - vma_results['f1']:+.4f}")
        print(f"  AP: {refined_results['ap'] - vma_results['ap']:+.4f}")


if __name__ == '__main__':
    main()
