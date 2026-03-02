#!/usr/bin/env python3
"""
评测新训练的模型（crop_radius=4.0m, 噪声2-10cm）
"""
import torch
import json
import numpy as np
import open3d as o3d
import os
import sys

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# 配置
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"

CONTEXT_RADIUS = 0.5  # 测试小范围采样
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0
NUM_SAMPLES = 20

def transform_line_coord_2d(line, matrix):
    line = np.array(line)
    x, y = line[:, 0], line[:, 1]
    pts = np.vstack([x, y, np.ones_like(x)])
    pts_trans = matrix @ pts
    return pts_trans[:2].T

def load_vma_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    detections_by_timestamp = {}
    for img_key, img_data in data.items():
        timestamp = img_key.split('/')[-1].replace('.jpg', '')
        pred_instances = img_data.get('pred_instances', [])
        detections = []
        for inst in pred_instances:
            if 'res_data' in inst:
                detections.append({
                    'points_2d': np.array(inst['res_data']),
                    'confidence': inst.get('confidence_level', 0.0)
                })
        if detections:
            detections_by_timestamp[timestamp] = detections
    return detections_by_timestamp

def load_annotations(annot_dir):
    annotations = {}
    for annot_file in os.listdir(annot_dir):
        if annot_file.endswith('.json'):
            timestamp = annot_file.replace('.json', '')
            with open(os.path.join(annot_dir, annot_file), 'r') as f:
                annot_data = json.load(f)
                gt_instances = []
                for instance in annot_data.get('instances', []):
                    position = instance.get('position', [])
                    category = instance.get('category', 'unknown')
                    if position and len(position) >= 2:
                        gt_instances.append({
                            'points_2d': np.array(position, dtype=np.float32),
                            'category': category
                        })
                annotations[timestamp] = {
                    'M_bev2world': np.array(annot_data['M_bev2world']),
                    'M_cropped_reverse': np.array(annot_data['M_cropped_reverse']),
                    'gt_instances': gt_instances
                }
    return annotations

def image_to_world(points_2d, M_cropped_reverse, M_bev2world):
    points_bev = transform_line_coord_2d(points_2d, M_cropped_reverse)
    points_world = transform_line_coord_2d(points_bev, M_bev2world)
    return points_world

def resample_line(line_points, num_points=32):
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

def weighted_sampling_context(context_points, line_points, num_samples, decay_scale):
    if len(context_points) == 0:
        return np.zeros((num_samples, 4))
    if len(context_points) <= num_samples:
        pad = np.zeros((num_samples - len(context_points), 4))
        return np.vstack([context_points, pad])
    dists = np.min(np.linalg.norm(
        context_points[:, None, :3] - line_points[None, :, :], axis=2
    ), axis=1)
    intensity = context_points[:, 3]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    intensity_weights = 0.5 + intensity_norm
    weights = np.exp(-dists / decay_scale) * intensity_weights
    weights = weights / weights.sum()
    indices = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    return context_points[indices]

def extract_context_around_line(pcd_points, line_world, radius, num_points):
    line_min = line_world.min(axis=0) - radius
    line_max = line_world.max(axis=0) + radius
    mask_bbox = (
        (pcd_points[:, 0] >= line_min[0]) & (pcd_points[:, 0] <= line_max[0]) &
        (pcd_points[:, 1] >= line_min[1]) & (pcd_points[:, 1] <= line_max[1]) &
        (pcd_points[:, 2] >= line_min[2]) & (pcd_points[:, 2] <= line_max[2])
    )
    candidate_points = pcd_points[mask_bbox]
    if len(candidate_points) == 0:
        center = line_world.mean(axis=0)
        dists_to_center = np.linalg.norm(pcd_points[:, :3] - center, axis=1)
        closest_idx = np.argmin(dists_to_center)
        candidate_points = pcd_points[closest_idx:closest_idx+1]
    if len(candidate_points) > 10000:
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
    return sampled_context, context_points

def chamfer_distance(line1, line2):
    """双向Chamfer距离"""
    if len(line1) == 0 or len(line2) == 0:
        return float('inf')
    dists_pred_to_gt = np.min(np.linalg.norm(
        line1[:, None, :2] - line2[None, :, :2], axis=2
    ), axis=1).mean()
    dists_gt_to_pred = np.min(np.linalg.norm(
        line2[:, None, :2] - line1[None, :, :2], axis=2
    ), axis=1).mean()
    return (dists_pred_to_gt + dists_gt_to_pred) / 2

def line_to_pointcloud_distance(line, pcd_points):
    if len(pcd_points) == 0:
        return float('inf')
    dists = np.min(np.linalg.norm(
        line[:, None, :3] - pcd_points[None, :, :3], axis=2
    ), axis=1)
    return dists.mean()

def find_closest_gt(vma_line, gt_instances, M_cropped_reverse, M_bev2world, max_distance=0.5):
    min_dist = float('inf')
    closest_gt = None
    closest_gt_world = None
    for gt_inst in gt_instances:
        if gt_inst['category'] != 'lane_line':
            continue
        gt_line_2d = gt_inst['points_2d']
        gt_line_world_2d = image_to_world(gt_line_2d, M_cropped_reverse, M_bev2world)
        gt_line_world = np.column_stack([gt_line_world_2d, np.zeros(len(gt_line_world_2d))])
        gt_line_resampled = resample_line(gt_line_world, NUM_LINE_POINTS)
        dist = chamfer_distance(vma_line, gt_line_resampled)
        if dist < min_dist and dist < max_distance:
            min_dist = dist
            closest_gt = gt_inst
            closest_gt_world = gt_line_resampled
    return closest_gt, closest_gt_world

print("="*60)
print("评测新模型（crop_radius=4.0m, 噪声2-10cm）")
print("="*60)

print("\n加载数据...")
vma_detections = load_vma_results(VMA_JSON)
annotations = load_annotations(ANNOT_DIR)
pcd = o3d.io.read_point_cloud(PCD_PATH)
pcd_points = np.asarray(pcd.points)
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    intensity = colors[:, 0] * 255
else:
    intensity = pcd_points[:, 2]
pcd_points_with_intensity = np.column_stack([pcd_points, intensity])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

print(f"\n加载模型: {MODEL_PATH}")
model = LineRefineNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

annot_timestamps = sorted([int(ts) for ts in annotations.keys()])
metrics = []
count = 0

print(f"\n开始评测（crop_radius={CONTEXT_RADIUS}m）...")

for timestamp, det_list in vma_detections.items():
    if count >= NUM_SAMPLES:
        break
    ts_int = int(timestamp)
    closest_annot_ts = min(annot_timestamps, key=lambda x: abs(x - ts_int))
    time_diff = abs(closest_annot_ts - ts_int)
    if time_diff > 250_000_000:
        continue
    annot = annotations[str(closest_annot_ts)]
    M_cropped_reverse = annot['M_cropped_reverse']
    M_bev2world = annot['M_bev2world']

    for det_idx, det in enumerate(det_list):
        if count >= NUM_SAMPLES:
            break

        points_2d = det['points_2d']
        line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
        line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])
        vma_line_resampled = resample_line(line_world, NUM_LINE_POINTS)

        context, context_all = extract_context_around_line(pcd_points_with_intensity, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

        center = vma_line_resampled.mean(axis=0)
        context_xyz = context[:, :3] - center
        context_intensity = context[:, 3:4]
        context_normalized = np.hstack([context_xyz, context_intensity])
        vma_line_centered = vma_line_resampled - center

        context_tensor = torch.from_numpy(context_normalized).float().unsqueeze(0).to(device)
        line_tensor = torch.from_numpy(vma_line_centered).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred_offsets_stack = model(context_tensor, line_tensor)
            pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

        refined_line = vma_line_resampled + pred_offset

        closest_gt, gt_line_world = find_closest_gt(vma_line_resampled, annot['gt_instances'], M_cropped_reverse, M_bev2world)

        if closest_gt is not None and gt_line_world is not None:
            vma_to_gt = chamfer_distance(vma_line_resampled, gt_line_world)
            refined_to_gt = chamfer_distance(refined_line, gt_line_world)
            vma_to_pcd = line_to_pointcloud_distance(vma_line_resampled, context_all)
            refined_to_pcd = line_to_pointcloud_distance(refined_line, context_all)

            metrics.append({
                'vma_to_gt': vma_to_gt,
                'refined_to_gt': refined_to_gt,
                'vma_to_pcd': vma_to_pcd,
                'refined_to_pcd': refined_to_pcd
            })
            count += 1
            print(f"样本 {count}: VMA→GT={vma_to_gt*100:.2f}cm, Refined→GT={refined_to_gt*100:.2f}cm")

vma_to_gt_avg = np.mean([m['vma_to_gt'] for m in metrics])
refined_to_gt_avg = np.mean([m['refined_to_gt'] for m in metrics])
vma_to_pcd_avg = np.mean([m['vma_to_pcd'] for m in metrics])
refined_to_pcd_avg = np.mean([m['refined_to_pcd'] for m in metrics])

print(f"\n{'='*60}")
print(f"评测结果总结")
print(f"{'='*60}")
print(f"\n到GT距离:")
print(f"  VMA平均: {vma_to_gt_avg:.4f}m ({vma_to_gt_avg*100:.2f}cm)")
print(f"  Refined平均: {refined_to_gt_avg:.4f}m ({refined_to_gt_avg*100:.2f}cm)")
improvement = vma_to_gt_avg - refined_to_gt_avg
improvement_pct = (improvement / vma_to_gt_avg * 100) if vma_to_gt_avg > 0 else 0
print(f"  改进: {improvement:.4f}m ({improvement*100:.2f}cm, {improvement_pct:.1f}%)")

print(f"\n到点云距离:")
print(f"  VMA平均: {vma_to_pcd_avg:.4f}m ({vma_to_pcd_avg*100:.2f}cm)")
print(f"  Refined平均: {refined_to_pcd_avg:.4f}m ({refined_to_pcd_avg*100:.2f}cm)")
pcd_improvement = vma_to_pcd_avg - refined_to_pcd_avg
pcd_improvement_pct = (pcd_improvement / vma_to_pcd_avg * 100) if vma_to_pcd_avg > 0 else 0
print(f"  改进: {pcd_improvement:.4f}m ({pcd_improvement*100:.2f}cm, {pcd_improvement_pct:.1f}%)")

result = {
    'model_path': MODEL_PATH,
    'crop_radius': CONTEXT_RADIUS,
    'vma_to_gt_avg': float(vma_to_gt_avg),
    'refined_to_gt_avg': float(refined_to_gt_avg),
    'vma_to_pcd_avg': float(vma_to_pcd_avg),
    'refined_to_pcd_avg': float(refined_to_pcd_avg),
    'gt_improvement': float(improvement),
    'gt_improvement_pct': float(improvement_pct),
    'pcd_improvement_pct': float(pcd_improvement_pct)
}

output_json = "/homes/zhangzijian/pointnet_refine/visualizations/new_model_evaluation.json"
with open(output_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n结果已保存到: {output_json}")
