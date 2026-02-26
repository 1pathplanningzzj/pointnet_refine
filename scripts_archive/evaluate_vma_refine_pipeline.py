#!/usr/bin/env python3
"""
完整的VMA检测结果Refine评估流程:
1. 根据VMA检测结果采样点云
2. 使用refine模型进行推理
3. 评估refine前后的精度提升
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys
from pathlib import Path
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# ==================== 配置 ====================
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
GT_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# 评估参数
MATCH_THRESHOLD = 3.0  # 匹配阈值(米)

# ==================== 工具函数 ====================

def transform_line_coord_2d(line, matrix):
    """2D坐标变换"""
    line = np.array(line)
    x, y = line[:, 0], line[:, 1]
    pts = np.vstack([x, y, np.ones_like(x)])
    pts_trans = matrix @ pts
    return pts_trans[:2].T

def load_vma_results(json_path):
    """加载VMA检测结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    detections_by_timestamp = {}
    for img_key, img_data in data.items():
        timestamp = img_key.split('/')[-1].replace('.jpg', '')
        detections_by_timestamp[timestamp] = []

        for inst in img_data['pred_instances']:
            detections_by_timestamp[timestamp].append({
                'class': inst['class'],
                'points_2d': np.array(inst['data']),
                'attrs': inst.get('attrs', {}),
                'confidence_level': inst.get('confidence_level', 0),
                'line_length_m': inst.get('line_length_m', 0)
            })

    print(f"加载了 {len(detections_by_timestamp)} 个时间戳的检测结果")
    total = sum(len(v) for v in detections_by_timestamp.values())
    print(f"总共 {total} 个检测实例")

    return detections_by_timestamp

def load_annotations(annot_dir):
    """加载annotation（转换矩阵）"""
    annotations = {}
    for annot_file in os.listdir(annot_dir):
        if annot_file.endswith('.json'):
            timestamp = annot_file.replace('.json', '')
            with open(os.path.join(annot_dir, annot_file), 'r') as f:
                annot_data = json.load(f)
                annotations[timestamp] = {
                    'M_bev2world': np.array(annot_data['M_bev2world']),
                    'M_cropped_reverse': np.array(annot_data['M_cropped_reverse'])
                }

    print(f"加载了 {len(annotations)} 个annotation")
    return annotations

def load_gt_lines(gt_json_path):
    """加载GT线"""
    with open(gt_json_path, 'r') as f:
        data = json.load(f)

    gt_lines = []
    if 'lane_lines' in data:
        for lane in data['lane_lines']:
            pts = lane.get('pts', [])
            if not pts or len(pts) < 2:
                continue
            xs = pts[0] if len(pts) > 0 else []
            ys = pts[1] if len(pts) > 1 else []
            zs = pts[2] if len(pts) > 2 else [0.0] * len(xs)
            n = min(len(xs), len(ys), len(zs) if zs else len(xs))
            if n < 2:
                continue
            points = np.array([[xs[i], ys[i], zs[i] if zs else 0.0] for i in range(n)], dtype=np.float32)
            gt_lines.append(points)
    elif 'items' in data:
        for item in data['items']:
            raw_pts = item.get('position', [])
            if raw_pts:
                pts = np.array([[p['x'], p['y'], p['z']] for p in raw_pts], dtype=np.float32)
                gt_lines.append(pts)

    print(f"加载了 {len(gt_lines)} 条GT线")
    return gt_lines

def image_to_world(points_2d, M_cropped_reverse, M_bev2world):
    """图像坐标 -> 世界坐标"""
    points_bev = transform_line_coord_2d(points_2d, M_cropped_reverse)
    points_world = transform_line_coord_2d(points_bev, M_bev2world)
    return points_world

def resample_line(line_points, num_points=32):
    """重采样线到固定点数"""
    if len(line_points) < 2:
        return np.tile(line_points, (num_points, 1))[:num_points]

    dists = np.sqrt(np.sum(np.diff(line_points, axis=0)**2, axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    if total_length < 1e-6:
        return np.tile(line_points[0:1], (num_points, 1))

    target_dists = np.linspace(0, total_length, num_points)
    resampled = np.zeros((num_points, line_points.shape[1]))
    for i in range(line_points.shape[1]):
        resampled[:, i] = np.interp(target_dists, cum_dists, line_points[:, i])

    return resampled

def weighted_sampling_context(context_points, line_points, num_samples=NUM_CONTEXT_POINTS, decay_scale=DECAY_SCALE):
    """加权采样context点云"""
    from scipy.spatial import KDTree

    if len(context_points) <= num_samples:
        if len(context_points) == 0:
            return np.zeros((num_samples, 4))
        indices = np.random.choice(len(context_points), num_samples, replace=True)
        return context_points[indices]

    tree = KDTree(line_points)
    distances, _ = tree.query(context_points[:, :3])
    dist_weights = np.exp(-distances / decay_scale)

    intensities = context_points[:, 3]
    inten_min = np.min(intensities)
    inten_max = np.max(intensities)
    if inten_max > inten_min:
        inten_norm = (intensities - inten_min) / (inten_max - inten_min + 1e-6)
    else:
        inten_norm = np.ones_like(intensities) * 0.5
    inten_weights = 0.5 + inten_norm

    weights = dist_weights * inten_weights
    w_sum = weights.sum()
    if w_sum < 1e-6:
        weights = None
    else:
        weights = weights / w_sum

    choice = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    return context_points[choice]

def extract_context_around_line(pcd_points, line_world, radius=CONTEXT_RADIUS, num_points=NUM_CONTEXT_POINTS):
    """提取线周围的点云 - 优化内存使用"""
    # 先用bounding box粗筛选，减少计算量
    line_min = line_world.min(axis=0) - radius
    line_max = line_world.max(axis=0) + radius

    mask_bbox = (
        (pcd_points[:, 0] >= line_min[0]) & (pcd_points[:, 0] <= line_max[0]) &
        (pcd_points[:, 1] >= line_min[1]) & (pcd_points[:, 1] <= line_max[1]) &
        (pcd_points[:, 2] >= line_min[2]) & (pcd_points[:, 2] <= line_max[2])
    )

    candidate_points = pcd_points[mask_bbox]

    if len(candidate_points) == 0:
        # 如果bbox内没有点，找最近的点
        center = line_world.mean(axis=0)
        dists_to_center = np.linalg.norm(pcd_points[:, :3] - center, axis=1)
        closest_idx = np.argmin(dists_to_center)
        candidate_points = pcd_points[closest_idx:closest_idx+1]

    # 在候选点中计算精确距离
    if len(candidate_points) > 10000:
        # 如果候选点太多，先随机采样减少计算量
        indices = np.random.choice(len(candidate_points), 10000, replace=False)
        candidate_points = candidate_points[indices]

    dists = np.min(np.linalg.norm(
        candidate_points[:, None, :3] - line_world[None, :, :], axis=2
    ), axis=1)

    mask = dists < radius
    context_points = candidate_points[mask]

    if len(context_points) == 0:
        closest_idx = np.argmin(dists)
        context_points = candidate_points[closest_idx:closest_idx+1]

    sampled_context = weighted_sampling_context(context_points, line_world, num_points, DECAY_SCALE)
    return sampled_context

def chamfer_distance(pred_line, gt_line):
    """计算Chamfer距离"""
    pred_to_gt = np.min(distance.cdist(pred_line[:, :2], gt_line[:, :2], 'euclidean'), axis=1)
    gt_to_pred = np.min(distance.cdist(gt_line[:, :2], pred_line[:, :2], 'euclidean'), axis=0)
    return np.mean(pred_to_gt) + np.mean(gt_to_pred)

def match_predictions_to_gt(pred_lines, gt_lines, threshold=MATCH_THRESHOLD):
    """匹配预测线到GT线"""
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return {}

    cost_matrix = np.full((len(pred_lines), len(gt_lines)), 1000.0, dtype=np.float32)
    for i, pred in enumerate(pred_lines):
        for j, gt in enumerate(gt_lines):
            cost_matrix[i, j] = chamfer_distance(pred, gt)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:
            matches[r] = c

    return matches

# ==================== Refine推理 ====================

def refine_vma_detections(vma_detections, annotations, pcd_points, model, device):
    """使用refine模型优化VMA检测结果"""
    refined_results = {}
    original_results = {}

    total_lines = sum(len(v) for v in vma_detections.values())
    processed = 0

    # 将annotation的时间戳转换为整数列表，用于最近邻匹配
    annot_timestamps = sorted([int(ts) for ts in annotations.keys()])

    for timestamp, det_list in vma_detections.items():
        # 找到最近的annotation时间戳
        ts_int = int(timestamp)
        closest_annot_ts = min(annot_timestamps, key=lambda x: abs(x - ts_int))
        time_diff = abs(closest_annot_ts - ts_int)

        # 如果时间差超过250ms，跳过
        if time_diff > 250_000_000:
            print(f"警告: 时间戳 {timestamp} 最近的annotation相差 {time_diff/1e6:.1f}ms，跳过")
            continue

        annot = annotations[str(closest_annot_ts)]
        M_cropped_reverse = annot['M_cropped_reverse']
        M_bev2world = annot['M_bev2world']

        refined_results[timestamp] = []
        original_results[timestamp] = []

        for det in det_list:
            # 1. 转换到世界坐标
            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])

            # 保存原始线
            original_results[timestamp].append(line_world)

            # 2. 重采样到32个点
            line_resampled = resample_line(line_world, NUM_LINE_POINTS)

            # 3. 提取周围点云
            context = extract_context_around_line(pcd_points, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

            # 4. 准备模型输入
            context_tensor = torch.from_numpy(context).float().unsqueeze(0).to(device)
            line_tensor = torch.from_numpy(line_resampled).float().unsqueeze(0).to(device)

            # 5. 模型推理
            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

            # 6. 计算refined线
            refined_line = line_resampled + pred_offset
            refined_results[timestamp].append(refined_line)

            processed += 1
            if processed % 50 == 0:
                print(f"已处理 {processed}/{total_lines} 条线")

    print(f"完成！共处理 {processed} 条线")
    return original_results, refined_results

# ==================== 评估 ====================

def evaluate_results(original_results, refined_results, gt_lines):
    """评估refine前后的精度"""
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)

    all_original_chamfer = []
    all_refined_chamfer = []
    improved_count = 0
    total_matched = 0

    for timestamp in original_results.keys():
        if timestamp not in refined_results:
            continue

        orig_lines = original_results[timestamp]
        ref_lines = refined_results[timestamp]

        # 匹配到GT
        orig_matches = match_predictions_to_gt(orig_lines, gt_lines)
        ref_matches = match_predictions_to_gt(ref_lines, gt_lines)

        # 计算匹配上的线的Chamfer距离
        for pred_idx in orig_matches.keys():
            if pred_idx not in ref_matches:
                continue

            gt_idx_orig = orig_matches[pred_idx]
            gt_idx_ref = ref_matches[pred_idx]

            # 只比较匹配到同一条GT的情况
            if gt_idx_orig != gt_idx_ref:
                continue

            gt_line = gt_lines[gt_idx_orig]
            orig_chamfer = chamfer_distance(orig_lines[pred_idx], gt_line)
            ref_chamfer = chamfer_distance(ref_lines[pred_idx], gt_line)

            all_original_chamfer.append(orig_chamfer)
            all_refined_chamfer.append(ref_chamfer)

            if ref_chamfer < orig_chamfer:
                improved_count += 1

            total_matched += 1

    if total_matched == 0:
        print("没有匹配到GT的线，无法评估")
        return

    # 统计结果
    avg_orig = np.mean(all_original_chamfer)
    avg_ref = np.mean(all_refined_chamfer)
    improvement = ((avg_orig - avg_ref) / avg_orig) * 100

    print(f"\n总共匹配到GT的线: {total_matched}")
    print(f"Refine后改善的线: {improved_count} ({improved_count/total_matched*100:.1f}%)")
    print(f"\n平均Chamfer距离:")
    print(f"  原始VMA: {avg_orig:.4f} m")
    print(f"  Refine后: {avg_ref:.4f} m")
    print(f"  改善: {improvement:.2f}%")

    # 中位数
    med_orig = np.median(all_original_chamfer)
    med_ref = np.median(all_refined_chamfer)
    print(f"\n中位数Chamfer距离:")
    print(f"  原始VMA: {med_orig:.4f} m")
    print(f"  Refine后: {med_ref:.4f} m")

    return {
        'total_matched': total_matched,
        'improved_count': improved_count,
        'avg_original': avg_orig,
        'avg_refined': avg_ref,
        'improvement_pct': improvement,
        'median_original': med_orig,
        'median_refined': med_ref
    }

# ==================== 主流程 ====================

def main():
    print("="*60)
    print("VMA检测结果Refine完整评估流程")
    print("="*60)

    # 1. 加载数据
    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations...")
    annotations = load_annotations(ANNOT_DIR)

    print("\n3. 加载GT线...")
    gt_lines = load_gt_lines(GT_JSON)

    print("\n4. 加载点云...")
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    pcd_points = np.asarray(pcd.points)

    # 添加intensity
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0] * 255
    else:
        intensity = pcd_points[:, 2]

    pcd_points_with_intensity = np.column_stack([pcd_points, intensity])
    print(f"点云: {len(pcd_points)} 个点")

    # 5. 加载模型
    print("\n5. 加载refine模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = LineRefineNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"成功加载模型: {MODEL_PATH}")
    else:
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return

    model.eval()

    # 6. Refine检测结果
    print("\n6. 开始refine推理...")
    original_results, refined_results = refine_vma_detections(
        vma_detections, annotations, pcd_points_with_intensity, model, device
    )

    # 7. 评估
    print("\n7. 评估refine效果...")
    eval_metrics = evaluate_results(original_results, refined_results, gt_lines)

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)

if __name__ == "__main__":
    main()
