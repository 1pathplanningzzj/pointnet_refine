#!/usr/bin/env python3
"""
可视化VMA检测结果、Refine结果和点云
用于检查坐标系对齐和refine效果
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
GT_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/vma_refine_vis"

# Refine参数 - 与训练时一致
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# 可视化参数
NUM_SAMPLES_TO_VIS = 5  # 可视化前N个样本

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
    """提取线周围的点云"""
    # 先用bounding box粗筛选
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
    return sampled_context

def create_line_geometry(points, color, radius=0.05):
    """创建线的几何体"""
    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    line_pcd.paint_uniform_color(color)
    return line_pcd

def visualize_sample(sample_idx, vma_line_resampled, refined_line, context_points, gt_lines, output_path):
    """可视化单个样本 - 保存为PLY文件"""
    geometries = []

    # 1. Context点云 (灰色)
    context_pcd = o3d.geometry.PointCloud()
    context_pcd.points = o3d.utility.Vector3dVector(context_points[:, :3])
    context_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(context_pcd)

    # 2. VMA检测线resampled (红色)
    vma_pcd = create_line_geometry(vma_line_resampled, [1.0, 0.0, 0.0], radius=0.08)
    geometries.append(vma_pcd)

    # 3. Refined线 (绿色)
    refined_pcd = create_line_geometry(refined_line, [0.0, 1.0, 0.0], radius=0.08)
    geometries.append(refined_pcd)

    # 4. GT线 (蓝色) - 只显示附近的GT
    for gt_line in gt_lines:
        # 检查GT线是否在context范围内
        center = vma_line_resampled.mean(axis=0)
        gt_center = gt_line.mean(axis=0)
        if np.linalg.norm(center[:2] - gt_center[:2]) < 20:  # 20米范围内
            gt_pcd = create_line_geometry(gt_line, [0.0, 0.0, 1.0], radius=0.08)
            geometries.append(gt_pcd)

    # 合并所有几何体并保存为PLY
    combined_pcd = o3d.geometry.PointCloud()
    for geom in geometries:
        combined_pcd += geom

    ply_path = output_path.replace('.png', '.ply')
    o3d.io.write_point_cloud(ply_path, combined_pcd)

    # 同时保存一个简单的文本说明
    txt_path = output_path.replace('.png', '.txt')
    offset_magnitude = np.linalg.norm(refined_line - vma_line_resampled, axis=1).mean()
    with open(txt_path, 'w') as f:
        f.write(f"Sample {sample_idx}\n")
        f.write(f"VMA line resampled (red): {len(vma_line_resampled)} points\n")
        f.write(f"Refined line (green): {len(refined_line)} points\n")
        f.write(f"Context points (gray): {len(context_points)} points\n")
        f.write(f"VMA center: {vma_line_resampled.mean(axis=0)}\n")
        f.write(f"Refined center: {refined_line.mean(axis=0)}\n")
        f.write(f"Average offset magnitude: {offset_magnitude:.4f} m\n")

    print(f"保存可视化到: {ply_path} (offset: {offset_magnitude:.4f}m)")

# ==================== 主流程 ====================

def main():
    print("="*60)
    print("VMA Refine结果可视化")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 6. 处理并可视化样本
    print(f"\n6. 处理并可视化前{NUM_SAMPLES_TO_VIS}个样本...")

    annot_timestamps = sorted([int(ts) for ts in annotations.keys()])
    vis_count = 0

    for timestamp, det_list in vma_detections.items():
        if vis_count >= NUM_SAMPLES_TO_VIS:
            break

        # 找到最近的annotation
        ts_int = int(timestamp)
        closest_annot_ts = min(annot_timestamps, key=lambda x: abs(x - ts_int))
        time_diff = abs(closest_annot_ts - ts_int)

        if time_diff > 250_000_000:
            continue

        annot = annotations[str(closest_annot_ts)]
        M_cropped_reverse = annot['M_cropped_reverse']
        M_bev2world = annot['M_bev2world']

        for det_idx, det in enumerate(det_list):
            if vis_count >= NUM_SAMPLES_TO_VIS:
                break

            # 转换到世界坐标
            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])

            # 重采样
            line_resampled = resample_line(line_world, NUM_LINE_POINTS)

            # 提取context
            context = extract_context_around_line(pcd_points_with_intensity, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

            # Refine推理
            context_tensor = torch.from_numpy(context).float().unsqueeze(0).to(device)
            line_tensor = torch.from_numpy(line_resampled).float().unsqueeze(0).to(device)

            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

            refined_line = line_resampled + pred_offset

            # 可视化
            output_path = os.path.join(OUTPUT_DIR, f"sample_{vis_count:03d}_ts{timestamp}_det{det_idx}.png")
            visualize_sample(vis_count, line_resampled, refined_line, context, gt_lines, output_path)

            vis_count += 1
            print(f"已可视化 {vis_count}/{NUM_SAMPLES_TO_VIS} 个样本")

    print(f"\n完成！可视化结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()