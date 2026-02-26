#!/usr/bin/env python3
"""
快速测试版本 - 只处理前10条线
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
OUTPUT_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_refined_epoch35_quick.json"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0
MAX_LINES = 10  # 只处理前10条线

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
        detections_by_timestamp[timestamp] = []

        for inst in img_data['pred_instances']:
            detections_by_timestamp[timestamp].append({
                'class': inst['class'],
                'points_2d': np.array(inst['data']),
                'line_length_m': inst.get('line_length_m', 0)
            })

    print(f"加载了 {len(detections_by_timestamp)} 个时间戳的检测结果")
    return detections_by_timestamp

def load_annotations(annot_dir):
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
    dists = np.min(np.linalg.norm(
        pcd_points[:, None, :3] - line_world[None, :, :], axis=2
    ), axis=1)

    mask = dists < radius
    context_points = pcd_points[mask]

    if len(context_points) == 0:
        closest_idx = np.argmin(dists)
        context_points = pcd_points[closest_idx:closest_idx+1]

    sampled_context = weighted_sampling_context(context_points, line_world, num_points, DECAY_SCALE)

    return sampled_context

def refine_vma_detections(vma_detections, annotations, pcd_points, model, device, max_lines=MAX_LINES):
    refined_results = {}
    processed = 0

    for timestamp, det_list in vma_detections.items():
        if processed >= max_lines:
            break

        if timestamp not in annotations:
            continue

        annot = annotations[timestamp]
        M_cropped_reverse = annot['M_cropped_reverse']
        M_bev2world = annot['M_bev2world']

        refined_results[timestamp] = []

        for det in det_list:
            if processed >= max_lines:
                break

            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])

            line_resampled = resample_line(line_world, NUM_LINE_POINTS)
            context = extract_context_around_line(pcd_points, line_world, CONTEXT_RADIUS, NUM_CONTEXT_POINTS)

            context_tensor = torch.from_numpy(context).float().unsqueeze(0).to(device)
            line_tensor = torch.from_numpy(line_resampled).float().unsqueeze(0).to(device)

            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

            refined_line = line_resampled + pred_offset

            refined_results[timestamp].append({
                'class': det['class'],
                'original_line_world': line_world.tolist(),
                'refined_line_world': refined_line.tolist(),
                'offset': pred_offset.tolist(),
                'line_length_m': det['line_length_m']
            })

            processed += 1
            print(f"已处理 {processed}/{max_lines} 条线")

    print(f"完成！共处理 {processed} 条线")
    return refined_results

def main():
    print("=" * 60)
    print("VMA检测结果Refine (快速测试版 - 前10条线)")
    print("=" * 60)

    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations...")
    annotations = load_annotations(ANNOT_DIR)

    print("\n3. 加载点云...")
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    pcd_points = np.asarray(pcd.points)

    intensity = pcd_points[:, 2]  # 使用Z作为intensity
    pcd_points_with_intensity = np.column_stack([pcd_points, intensity])
    print(f"点云: {len(pcd_points)} 个点")

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

    print("\n5. 开始refine (只处理前10条线)...")
    refined_results = refine_vma_detections(
        vma_detections, annotations, pcd_points_with_intensity, model, device, MAX_LINES
    )

    print(f"\n6. 保存refined结果到: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(refined_results, f, indent=2)

    print("\n完成！")
    print(f"Refined结果已保存到: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
