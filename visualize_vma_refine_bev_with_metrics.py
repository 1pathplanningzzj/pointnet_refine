#!/usr/bin/env python3
"""
VMA Refine结果的BEV可视化和精度评估
1. 生成BEV视角的可视化图像
2. 计算VMA到GT和Refined到GT的距离，评估精度提升
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# ==================== 配置 ====================
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based_0226/best_model.pth"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/vma_refine_bev_new_model"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# BEV参数
BEV_RESOLUTION = 0.02  # 2cm/pixel，提高分辨率
BEV_PADDING = 2.0  # 2m padding，减小padding来放大显示

# 可视化参数
NUM_SAMPLES_TO_VIS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 工具函数 ====================

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
                    # 使用position字段（图像坐标的分段GT线）
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

def weighted_sampling_context(context_points, line_points, num_samples=NUM_CONTEXT_POINTS, decay_scale=DECAY_SCALE):
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
    # 修正：和训练时保持一致，intensity权重范围[0.5, 1.5]
    intensity_weights = 0.5 + intensity_norm
    weights = np.exp(-dists / decay_scale) * intensity_weights
    weights = weights / weights.sum()
    indices = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    return context_points[indices]

def extract_context_around_line(pcd_points, line_world, radius=CONTEXT_RADIUS, num_points=NUM_CONTEXT_POINTS):
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

def chamfer_distance(line1, line2):
    """
    双向Chamfer距离，与trunk_line_dataset评测对齐
    line1: pred线 (N, 3)
    line2: gt线 (M, 3)
    返回: (pred→gt + gt→pred) / 2
    """
    if len(line1) == 0 or len(line2) == 0:
        return float('inf')

    # pred→gt: 对每个pred点找最近的gt点
    dists_pred_to_gt = np.min(np.linalg.norm(
        line1[:, None, :2] - line2[None, :, :2], axis=2
    ), axis=1).mean()

    # gt→pred: 对每个gt点找最近的pred点
    dists_gt_to_pred = np.min(np.linalg.norm(
        line2[:, None, :2] - line1[None, :, :2], axis=2
    ), axis=1).mean()

    # 双向平均
    return (dists_pred_to_gt + dists_gt_to_pred) / 2

def find_closest_gt(vma_line, gt_instances, M_cropped_reverse, M_bev2world, max_distance=0.5):
    """找到与VMA检测最接近的GT实例（GT在图像坐标，需要转换到世界坐标）"""
    min_dist = float('inf')
    closest_gt = None
    closest_gt_world = None

    for gt_inst in gt_instances:
        if gt_inst['category'] != 'lane_line':
            continue

        # 将GT从图像坐标转换到世界坐标
        gt_line_2d = gt_inst['points_2d']
        gt_line_world_2d = image_to_world(gt_line_2d, M_cropped_reverse, M_bev2world)
        gt_line_world = np.column_stack([gt_line_world_2d, np.zeros(len(gt_line_world_2d))])

        # 重采样GT到32个点
        gt_line_resampled = resample_line(gt_line_world, NUM_LINE_POINTS)

        # 计算VMA到GT的单向距离
        dist = chamfer_distance(vma_line, gt_line_resampled)

        if dist < min_dist and dist < max_distance:
            min_dist = dist
            closest_gt = gt_inst
            closest_gt_world = gt_line_resampled

    return closest_gt, closest_gt_world

def generate_bev_image(pcd_points, vma_line, refined_line, gt_line, resolution=BEV_RESOLUTION, padding=BEV_PADDING):
    """生成BEV视角的可视化图像"""
    # 合并所有点来确定范围
    all_points = [pcd_points[:, :2]]
    if vma_line is not None:
        all_points.append(vma_line[:, :2])
    if refined_line is not None:
        all_points.append(refined_line[:, :2])
    if gt_line is not None:
        all_points.append(gt_line[:, :2])

    all_points = np.vstack(all_points)
    x_min, y_min = all_points.min(axis=0) - padding
    x_max, y_max = all_points.max(axis=0) + padding

    # 创建BEV图像
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    # 创建intensity图
    bev_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制点云intensity
    x_idx = ((pcd_points[:, 0] - x_min) / resolution).astype(int)
    y_idx = ((pcd_points[:, 1] - y_min) / resolution).astype(int)
    valid_mask = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)

    intensity = pcd_points[:, 3]
    intensity_norm = np.clip((intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6), 0, 1)

    for i in np.where(valid_mask)[0]:
        color_val = int(intensity_norm[i] * 255)
        bev_img[y_idx[i], x_idx[i]] = [color_val, color_val, color_val]

    # 创建matplotlib figure - 增大尺寸和DPI
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(bev_img, origin='lower', extent=[x_min, x_max, y_min, y_max])

    # 绘制线条 - 再次调细线宽
    if gt_line is not None:
        ax.plot(gt_line[:, 0], gt_line[:, 1], 'c-', linewidth=1, label='GT', alpha=0.9)
        ax.scatter(gt_line[:, 0], gt_line[:, 1], c='cyan', s=8, alpha=0.9, zorder=5)
    if vma_line is not None:
        ax.plot(vma_line[:, 0], vma_line[:, 1], 'r-', linewidth=1, label='VMA', alpha=0.9)
        ax.scatter(vma_line[:, 0], vma_line[:, 1], c='red', s=6, alpha=0.9, zorder=5)
    if refined_line is not None:
        ax.plot(refined_line[:, 0], refined_line[:, 1], 'g-', linewidth=1, label='Refined', alpha=0.9)
        ax.scatter(refined_line[:, 0], refined_line[:, 1], c='lime', s=6, alpha=0.9, zorder=5)

    ax.legend(fontsize=14, loc='upper right')
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=12)

    return fig

def main():
    print("="*60)
    print("VMA Refine结果BEV可视化和精度评估")
    print("="*60)

    # 1. 加载数据
    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations (包含GT)...")
    annotations = load_annotations(ANNOT_DIR)
    print(f"加载了 {len(annotations)} 个annotation文件")

    print("\n3. 加载点云...")
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    pcd_points = np.asarray(pcd.points)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0] * 255
    else:
        intensity = pcd_points[:, 2]

    pcd_points_with_intensity = np.column_stack([pcd_points, intensity])
    print(f"点云: {len(pcd_points)} 个点")

    # 4. 加载模型
    print("\n4. 加载refine模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = LineRefineNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 5. 处理并评估
    print(f"\n5. 处理并生成BEV可视化（前{NUM_SAMPLES_TO_VIS}个样本）...")

    annot_timestamps = sorted([int(ts) for ts in annotations.keys()])
    vis_count = 0
    metrics = []

    for timestamp, det_list in vma_detections.items():
        if vis_count >= NUM_SAMPLES_TO_VIS:
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
            if vis_count >= NUM_SAMPLES_TO_VIS:
                break

            # 转换到世界坐标
            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])

            # 重采样
            vma_line_resampled = resample_line(line_world, NUM_LINE_POINTS)

            # 提取context
            context = extract_context_around_line(pcd_points_with_intensity, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

            # ===== 关键：中心化（和训练时一致）=====
            center = vma_line_resampled.mean(axis=0)
            context_xyz = context[:, :3] - center
            context_intensity = context[:, 3:4]
            context_normalized = np.hstack([context_xyz, context_intensity])
            vma_line_centered = vma_line_resampled - center

            # Refine推理
            context_tensor = torch.from_numpy(context_normalized).float().unsqueeze(0).to(device)
            line_tensor = torch.from_numpy(vma_line_centered).float().unsqueeze(0).to(device)

            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

            # offset是在centered坐标系下的，直接加到原始坐标上
            refined_line = vma_line_resampled + pred_offset

            # 找到最近的GT（使用position字段并转换到世界坐标）
            closest_gt, gt_line_world = find_closest_gt(vma_line_resampled, annot['gt_instances'], M_cropped_reverse, M_bev2world)

            if closest_gt is not None and gt_line_world is not None:
                # 计算距离（VMA/Refined到GT的单向距离）
                vma_to_gt = chamfer_distance(vma_line_resampled, gt_line_world)
                refined_to_gt = chamfer_distance(refined_line, gt_line_world)
                improvement = vma_to_gt - refined_to_gt
                improvement_pct = (improvement / vma_to_gt) * 100 if vma_to_gt > 0 else 0

                metrics.append({
                    'vma_to_gt': vma_to_gt,
                    'refined_to_gt': refined_to_gt,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                })

                # 生成BEV可视化
                fig = generate_bev_image(context, vma_line_resampled, refined_line, gt_line_world)
                fig.suptitle(f'Sample {vis_count} - VMA→GT: {vma_to_gt:.3f}m, Refined→GT: {refined_to_gt:.3f}m, Improvement: {improvement_pct:.1f}%', fontsize=14)

                output_path = os.path.join(OUTPUT_DIR, f"sample_{vis_count:03d}_ts{timestamp}_det{det_idx}.png")
                fig.savefig(output_path, dpi=200, bbox_inches='tight')  # 提高DPI到200
                plt.close(fig)

                print(f"  样本 {vis_count}: VMA→GT={vma_to_gt:.3f}m, Refined→GT={refined_to_gt:.3f}m, 提升={improvement_pct:.1f}%")
                vis_count += 1

    # 6. 统计结果
    print(f"\n{'='*60}")
    print("精度评估统计")
    print(f"{'='*60}")

    if metrics:
        vma_to_gt_avg = np.mean([m['vma_to_gt'] for m in metrics])
        refined_to_gt_avg = np.mean([m['refined_to_gt'] for m in metrics])
        improvement_avg = np.mean([m['improvement'] for m in metrics])
        improvement_pct_avg = np.mean([m['improvement_pct'] for m in metrics])

        print(f"样本数量: {len(metrics)}")
        print(f"VMA到GT平均距离: {vma_to_gt_avg:.4f}m")
        print(f"Refined到GT平均距离: {refined_to_gt_avg:.4f}m")
        print(f"平均改进: {improvement_avg:.4f}m ({improvement_pct_avg:.2f}%)")

        # 保存metrics到JSON
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'summary': {
                    'num_samples': len(metrics),
                    'vma_to_gt_avg': float(vma_to_gt_avg),
                    'refined_to_gt_avg': float(refined_to_gt_avg),
                    'improvement_avg': float(improvement_avg),
                    'improvement_pct_avg': float(improvement_pct_avg)
                },
                'details': metrics
            }, f, indent=2)
        print(f"\n指标已保存到: {metrics_path}")

    print(f"\n完成！BEV可视化结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

