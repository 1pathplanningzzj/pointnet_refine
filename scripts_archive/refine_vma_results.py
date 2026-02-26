#!/usr/bin/env python3
"""
使用refine模型优化VMA检测结果
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# 配置
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/cropped_data/annots"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_refined_epoch35.json"

# Refine参数 - 必须与训练时一致！
CONTEXT_RADIUS = 4.0  # 提取线周围4米的点云作为context (训练时crop_radius=4.0)
NUM_CONTEXT_POINTS = 2048  # 采样到2048个点 (训练时num_context_points=2048)
NUM_LINE_POINTS = 32  # 重采样线到32个点
DECAY_SCALE = 2.0  # 加权采样的衰减尺度 (训练时decay_scale=2.0)

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

def image_to_world(points_2d, M_cropped_reverse, M_bev2world):
    """图像坐标 -> 世界坐标"""
    points_bev = transform_line_coord_2d(points_2d, M_cropped_reverse)
    points_world = transform_line_coord_2d(points_bev, M_bev2world)
    return points_world

def resample_line(line_points, num_points=32):
    """
    重采样线到固定点数
    使用线性插值
    """
    if len(line_points) < 2:
        # 如果点太少，复制点
        return np.tile(line_points, (num_points, 1))[:num_points]

    # 计算累积距离
    dists = np.sqrt(np.sum(np.diff(line_points, axis=0)**2, axis=1))
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    if total_length < 1e-6:
        return np.tile(line_points[0:1], (num_points, 1))

    # 均匀采样
    target_dists = np.linspace(0, total_length, num_points)

    # 插值
    resampled = np.zeros((num_points, line_points.shape[1]))
    for i in range(line_points.shape[1]):
        resampled[:, i] = np.interp(target_dists, cum_dists, line_points[:, i])

    return resampled

def weighted_sampling_context(context_points, line_points, num_samples=NUM_CONTEXT_POINTS, decay_scale=DECAY_SCALE):
    """
    加权采样 - 与训练时保持一致
    基于：
    1. 距离线的距离（越近权重越大）
    2. Intensity（越高权重越大）
    """
    from scipy.spatial import KDTree

    if len(context_points) <= num_samples:
        if len(context_points) == 0:
            return np.zeros((num_samples, 4))
        # 重复采样填充
        indices = np.random.choice(len(context_points), num_samples, replace=True)
        return context_points[indices]

    # 计算到线的距离
    tree = KDTree(line_points)
    distances, _ = tree.query(context_points[:, :3])

    # 1. 距离权重
    dist_weights = np.exp(-distances / decay_scale)

    # 2. Intensity权重
    intensities = context_points[:, 3]
    inten_min = np.min(intensities)
    inten_max = np.max(intensities)
    if inten_max > inten_min:
        inten_norm = (intensities - inten_min) / (inten_max - inten_min + 1e-6)
    else:
        inten_norm = np.ones_like(intensities) * 0.5
    inten_weights = 0.5 + inten_norm  # [0.5, 1.5]

    # 组合权重
    weights = dist_weights * inten_weights

    w_sum = weights.sum()
    if w_sum < 1e-6:
        weights = None
    else:
        weights = weights / w_sum

    choice = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    return context_points[choice]

def extract_context_around_line(pcd_points, line_world, radius=CONTEXT_RADIUS, num_points=NUM_CONTEXT_POINTS):
    """
    提取线周围的点云作为context - 使用加权采样
    """
    # 计算每个点云点到线的最小距离
    dists = np.min(np.linalg.norm(
        pcd_points[:, None, :3] - line_world[None, :, :], axis=2
    ), axis=1)

    # 筛选半径内的点
    mask = dists < radius
    context_points = pcd_points[mask]

    if len(context_points) == 0:
        # 如果没有点，返回最近的点
        closest_idx = np.argmin(dists)
        context_points = pcd_points[closest_idx:closest_idx+1]

    # 使用加权采样（与训练时一致）
    sampled_context = weighted_sampling_context(context_points, line_world, num_points, DECAY_SCALE)

    return sampled_context

def refine_vma_detections(vma_detections, annotations, pcd_points, model, device):
    """
    使用refine模型优化VMA检测结果
    """
    refined_results = {}

    total_lines = sum(len(v) for v in vma_detections.values())
    processed = 0

    for timestamp, det_list in vma_detections.items():
        if timestamp not in annotations:
            print(f"警告: 时间戳 {timestamp} 没有annotation，跳过")
            continue

        annot = annotations[timestamp]
        M_cropped_reverse = annot['M_cropped_reverse']
        M_bev2world = annot['M_bev2world']

        refined_results[timestamp] = []

        for det in det_list:
            # 1. 转换到世界坐标
            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)

            # 添加Z坐标（假设在地面上，Z=0）
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])

            # 2. 重采样到32个点
            line_resampled = resample_line(line_world, NUM_LINE_POINTS)

            # 3. 提取周围点云
            context = extract_context_around_line(pcd_points, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

            # 4. 准备模型输入
            context_tensor = torch.from_numpy(context).float().unsqueeze(0).to(device)  # (1, N, 4)
            line_tensor = torch.from_numpy(line_resampled).float().unsqueeze(0).to(device)  # (1, M, 3)

            # 5. 模型推理
            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)  # (L, 1, M, 3)
                # 使用最后一层的输出
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()  # (M, 3)

            # 6. 计算refined线
            refined_line = line_resampled + pred_offset

            # 7. 保存结果
            refined_results[timestamp].append({
                'class': det['class'],
                'original_line_world': line_world.tolist(),
                'refined_line_world': refined_line.tolist(),
                'offset': pred_offset.tolist(),
                'attrs': det['attrs'],
                'confidence_level': det['confidence_level'],
                'line_length_m': det['line_length_m']
            })

            processed += 1
            if processed % 10 == 0:
                print(f"已处理 {processed}/{total_lines} 条线")

    print(f"完成！共处理 {processed} 条线")
    return refined_results

def main():
    print("=" * 60)
    print("VMA检测结果Refine")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations...")
    annotations = load_annotations(ANNOT_DIR)

    print("\n3. 加载点云...")
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    pcd_points = np.asarray(pcd.points)

    # 添加intensity（如果有）
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0] * 255  # 假设intensity存在颜色通道
    else:
        # 如果没有intensity，使用Z坐标作为替代
        intensity = pcd_points[:, 2]

    pcd_points_with_intensity = np.column_stack([pcd_points, intensity])
    print(f"点云: {len(pcd_points)} 个点")

    # 4. 加载模型
    print("\n4. 加载refine模型...")
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

    # 5. Refine检测结果
    print("\n5. 开始refine...")
    refined_results = refine_vma_detections(
        vma_detections, annotations, pcd_points_with_intensity, model, device
    )

    # 6. 保存结果
    print(f"\n6. 保存refined结果到: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(refined_results, f, indent=2)

    print("\n完成！")
    print(f"Refined结果已保存到: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
