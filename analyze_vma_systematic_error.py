#!/usr/bin/env python3
"""
分析 VMA 检测误差是否有系统性偏差
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/results_fuse175_test316.json"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/vma_error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        gt_line_resampled = resample_line(gt_line_world, 32)

        # Chamfer distance
        dists_pred_to_gt = np.min(np.linalg.norm(
            vma_line[:, None, :2] - gt_line_resampled[None, :, :2], axis=2
        ), axis=1).mean()
        dists_gt_to_pred = np.min(np.linalg.norm(
            gt_line_resampled[:, None, :2] - vma_line[None, :, :2], axis=2
        ), axis=1).mean()
        dist = (dists_pred_to_gt + dists_gt_to_pred) / 2

        if dist < min_dist and dist < max_distance:
            min_dist = dist
            closest_gt = gt_inst
            closest_gt_world = gt_line_resampled
    return closest_gt, closest_gt_world

print("加载数据...")
vma_detections = load_vma_results(VMA_JSON)
annotations = load_annotations(ANNOT_DIR)

print("分析 VMA 误差...")
annot_timestamps = sorted([int(ts) for ts in annotations.keys()])

# 存储误差向量
error_vectors = []  # (dx, dy) for each point
error_magnitudes = []
lateral_errors = []  # 横向误差（垂直于线方向）
longitudinal_errors = []  # 纵向误差（沿线方向）

count = 0
max_samples = 100

for timestamp, det_list in vma_detections.items():
    if count >= max_samples:
        break

    ts_int = int(timestamp)
    closest_annot_ts = min(annot_timestamps, key=lambda x: abs(x - ts_int))
    time_diff = abs(closest_annot_ts - ts_int)

    if time_diff > 250_000_000:
        continue

    annot = annotations[str(closest_annot_ts)]
    M_cropped_reverse = annot['M_cropped_reverse']
    M_bev2world = annot['M_bev2world']

    for det in det_list:
        if count >= max_samples:
            break

        points_2d = det['points_2d']
        line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
        vma_line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])
        vma_line_resampled = resample_line(vma_line_world, 32)

        closest_gt, gt_line_world = find_closest_gt(
            vma_line_resampled, annot['gt_instances'],
            M_cropped_reverse, M_bev2world
        )

        if closest_gt is not None and gt_line_world is not None:
            # 计算每个点的误差向量
            for i in range(len(vma_line_resampled)):
                vma_pt = vma_line_resampled[i, :2]
                gt_pt = gt_line_world[i, :2]
                error_vec = vma_pt - gt_pt  # (dx, dy)
                error_vectors.append(error_vec)
                error_magnitudes.append(np.linalg.norm(error_vec))

                # 计算横向和纵向误差
                if i < len(gt_line_world) - 1:
                    # 线的方向向量
                    line_dir = gt_line_world[i+1, :2] - gt_line_world[i, :2]
                    line_dir_norm = line_dir / (np.linalg.norm(line_dir) + 1e-6)

                    # 纵向误差（沿线方向）
                    long_err = np.dot(error_vec, line_dir_norm)
                    longitudinal_errors.append(long_err)

                    # 横向误差（垂直于线方向）
                    lateral_dir = np.array([-line_dir_norm[1], line_dir_norm[0]])
                    lat_err = np.dot(error_vec, lateral_dir)
                    lateral_errors.append(lat_err)

            count += 1

error_vectors = np.array(error_vectors)
error_magnitudes = np.array(error_magnitudes)
lateral_errors = np.array(lateral_errors)
longitudinal_errors = np.array(longitudinal_errors)

print(f"\n分析了 {count} 条车道线，共 {len(error_vectors)} 个点")
print(f"\n误差统计:")
print(f"  平均误差幅度: {error_magnitudes.mean()*100:.2f}cm")
print(f"  误差标准差: {error_magnitudes.std()*100:.2f}cm")
print(f"\n误差向量统计:")
print(f"  平均 dx: {error_vectors[:, 0].mean()*100:.2f}cm (正=向东偏)")
print(f"  平均 dy: {error_vectors[:, 1].mean()*100:.2f}cm (正=向北偏)")
print(f"  dx 标准差: {error_vectors[:, 0].std()*100:.2f}cm")
print(f"  dy 标准差: {error_vectors[:, 1].std()*100:.2f}cm")
print(f"\n横向/纵向误差:")
print(f"  平均横向误差: {lateral_errors.mean()*100:.2f}cm (正=向左偏)")
print(f"  平均纵向误差: {longitudinal_errors.mean()*100:.2f}cm (正=向前偏)")
print(f"  横向误差标准差: {lateral_errors.std()*100:.2f}cm")
print(f"  纵向误差标准差: {longitudinal_errors.std()*100:.2f}cm")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 误差向量散点图
ax = axes[0, 0]
ax.scatter(error_vectors[:, 0]*100, error_vectors[:, 1]*100, alpha=0.3, s=1)
ax.axhline(0, color='r', linestyle='--', linewidth=0.5)
ax.axvline(0, color='r', linestyle='--', linewidth=0.5)
ax.set_xlabel('dx (cm, 东向)')
ax.set_ylabel('dy (cm, 北向)')
ax.set_title('VMA 误差向量分布')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 2. dx 直方图
ax = axes[0, 1]
ax.hist(error_vectors[:, 0]*100, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(error_vectors[:, 0].mean()*100, color='r', linestyle='--',
           label=f'Mean: {error_vectors[:, 0].mean()*100:.2f}cm')
ax.set_xlabel('dx (cm)')
ax.set_ylabel('频数')
ax.set_title('X 方向误差分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. dy 直方图
ax = axes[0, 2]
ax.hist(error_vectors[:, 1]*100, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(error_vectors[:, 1].mean()*100, color='r', linestyle='--',
           label=f'Mean: {error_vectors[:, 1].mean()*100:.2f}cm')
ax.set_xlabel('dy (cm)')
ax.set_ylabel('频数')
ax.set_title('Y 方向误差分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 横向误差分布
ax = axes[1, 0]
ax.hist(lateral_errors*100, bins=50, alpha=0.7, edgecolor='black', color='green')
ax.axvline(lateral_errors.mean()*100, color='r', linestyle='--',
           label=f'Mean: {lateral_errors.mean()*100:.2f}cm')
ax.set_xlabel('横向误差 (cm, 正=左偏)')
ax.set_ylabel('频数')
ax.set_title('横向误差分布（垂直于线）')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 纵向误差分布
ax = axes[1, 1]
ax.hist(longitudinal_errors*100, bins=50, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(longitudinal_errors.mean()*100, color='r', linestyle='--',
           label=f'Mean: {longitudinal_errors.mean()*100:.2f}cm')
ax.set_xlabel('纵向误差 (cm, 正=前偏)')
ax.set_ylabel('频数')
ax.set_title('纵向误差分布（沿线方向）')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 误差幅度分布
ax = axes[1, 2]
ax.hist(error_magnitudes*100, bins=50, alpha=0.7, edgecolor='black', color='purple')
ax.axvline(error_magnitudes.mean()*100, color='r', linestyle='--',
           label=f'Mean: {error_magnitudes.mean()*100:.2f}cm')
ax.set_xlabel('误差幅度 (cm)')
ax.set_ylabel('频数')
ax.set_title('误差幅度分布')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'vma_error_analysis.png'), dpi=150)
print(f"\n可视化已保存到: {OUTPUT_DIR}/vma_error_analysis.png")

# 判断是否有系统误差
print(f"\n系统误差判断:")
dx_mean = error_vectors[:, 0].mean() * 100
dy_mean = error_vectors[:, 1].mean() * 100
dx_std = error_vectors[:, 0].std() * 100
dy_std = error_vectors[:, 1].std() * 100

if abs(dx_mean) > dx_std * 0.3:
    print(f"  ⚠️  X 方向有系统误差: {dx_mean:.2f}cm (标准差的 {abs(dx_mean)/dx_std:.1f} 倍)")
else:
    print(f"  ✓ X 方向无明显系统误差")

if abs(dy_mean) > dy_std * 0.3:
    print(f"  ⚠️  Y 方向有系统误差: {dy_mean:.2f}cm (标准差的 {abs(dy_mean)/dy_std:.1f} 倍)")
else:
    print(f"  ✓ Y 方向无明显系统误差")

lat_mean = lateral_errors.mean() * 100
lat_std = lateral_errors.std() * 100
if abs(lat_mean) > lat_std * 0.3:
    print(f"  ⚠️  横向有系统误差: {lat_mean:.2f}cm (标准差的 {abs(lat_mean)/lat_std:.1f} 倍)")
else:
    print(f"  ✓ 横向无明显系统误差")