#!/usr/bin/env python3
"""
可视化VMA结果采样点云并进行推理的完整过程
展示：
1. VMA检测线（图像坐标 -> 世界坐标）
2. 点云context提取（bbox + radius filtering）
3. 加权采样（距离衰减 + intensity权重）
4. 中心化（centering）
5. 推理结果（offset + 恢复到世界坐标）
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys
import plotly.graph_objects as go
from pathlib import Path

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# ==================== 配置 ====================
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/sampling_process"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# 可视化参数
NUM_SAMPLES_TO_VIS = 5

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
                annotations[timestamp] = {
                    'M_bev2world': np.array(annot_data['M_bev2world']),
                    'M_cropped_reverse': np.array(annot_data['M_cropped_reverse'])
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

def extract_context_with_steps(pcd_points, line_world, radius=CONTEXT_RADIUS):
    """提取context，返回每个步骤的中间结果用于可视化"""
    steps = {}

    # Step 1: Bounding box filtering
    line_min = line_world.min(axis=0) - radius
    line_max = line_world.max(axis=0) + radius
    mask_bbox = (
        (pcd_points[:, 0] >= line_min[0]) & (pcd_points[:, 0] <= line_max[0]) &
        (pcd_points[:, 1] >= line_min[1]) & (pcd_points[:, 1] <= line_max[1]) &
        (pcd_points[:, 2] >= line_min[2]) & (pcd_points[:, 2] <= line_max[2])
    )
    bbox_points = pcd_points[mask_bbox]
    steps['bbox_points'] = bbox_points
    steps['bbox_min'] = line_min
    steps['bbox_max'] = line_max

    if len(bbox_points) == 0:
        center = line_world.mean(axis=0)
        dists_to_center = np.linalg.norm(pcd_points[:, :3] - center, axis=1)
        closest_idx = np.argmin(dists_to_center)
        bbox_points = pcd_points[closest_idx:closest_idx+1]
        steps['bbox_points'] = bbox_points

    # 如果bbox内点太多，先随机采样
    if len(bbox_points) > 10000:
        indices = np.random.choice(len(bbox_points), 10000, replace=False)
        bbox_points = bbox_points[indices]
        steps['bbox_points'] = bbox_points

    # Step 2: Radius filtering
    dists = np.min(np.linalg.norm(
        bbox_points[:, None, :3] - line_world[None, :, :], axis=2
    ), axis=1)
    mask_radius = dists < radius
    radius_points = bbox_points[mask_radius]
    steps['radius_points'] = radius_points
    steps['distances'] = dists[mask_radius]

    if len(radius_points) == 0:
        closest_idx = np.argmin(dists)
        radius_points = bbox_points[closest_idx:closest_idx+1]
        steps['radius_points'] = radius_points
        steps['distances'] = np.array([dists[closest_idx]])

    return steps

def weighted_sampling_with_weights(context_points, line_points, num_samples=NUM_CONTEXT_POINTS, decay_scale=DECAY_SCALE):
    """加权采样，返回采样结果和权重用于可视化"""
    if len(context_points) == 0:
        return np.zeros((num_samples, 4)), np.zeros(num_samples)

    if len(context_points) <= num_samples:
        pad = np.zeros((num_samples - len(context_points), 4))
        sampled = np.vstack([context_points, pad])
        weights = np.ones(len(context_points))
        weights = np.concatenate([weights, np.zeros(num_samples - len(context_points))])
        return sampled, weights

    # 计算距离权重
    dists = np.min(np.linalg.norm(
        context_points[:, None, :3] - line_points[None, :, :], axis=2
    ), axis=1)

    # 计算intensity权重（修正：和训练时保持一致，范围[0.5, 1.5]）
    intensity = context_points[:, 3]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    intensity_weights = 0.5 + intensity_norm

    # 综合权重
    weights = np.exp(-dists / decay_scale) * intensity_weights
    weights = weights / weights.sum()

    # 采样
    indices = np.random.choice(len(context_points), num_samples, replace=False, p=weights)
    sampled = context_points[indices]
    sampled_weights = weights[indices]

    return sampled, sampled_weights, weights, indices

def create_visualization(sample_idx, vma_line_original, vma_line_resampled,
                        context_steps, sampled_context, sampled_weights,
                        vma_line_centered, refined_line_centered, refined_line_final,
                        center):
    """创建完整的采样和推理过程可视化"""

    fig = go.Figure()

    # 1. 原始点云（bbox内的点）
    bbox_points = context_steps['bbox_points']
    if len(bbox_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=bbox_points[:, 0], y=bbox_points[:, 1], z=bbox_points[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.3),
            name='BBox内点云',
            hovertemplate='BBox点<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))

    # 2. Radius过滤后的点云
    radius_points = context_steps['radius_points']
    if len(radius_points) > 0:
        distances = context_steps['distances']
        fig.add_trace(go.Scatter3d(
            x=radius_points[:, 0], y=radius_points[:, 1], z=radius_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=distances,
                colorscale='Viridis',
                colorbar=dict(title='距离VMA线(m)', x=1.15),
                opacity=0.6
            ),
            name='Radius过滤点云',
            hovertemplate='Radius点<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>距离: %{marker.color:.2f}m<extra></extra>'
        ))

    # 3. 加权采样后的点云（用权重着色）
    if len(sampled_context) > 0:
        fig.add_trace(go.Scatter3d(
            x=sampled_context[:, 0], y=sampled_context[:, 1], z=sampled_context[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=sampled_weights,
                colorscale='Hot',
                colorbar=dict(title='采样权重', x=1.3),
                opacity=0.9
            ),
            name='采样后点云(2048点)',
            hovertemplate='采样点<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>权重: %{marker.color:.4f}<extra></extra>'
        ))

    # 4. VMA原始线（图像坐标转换后）
    fig.add_trace(go.Scatter3d(
        x=vma_line_original[:, 0], y=vma_line_original[:, 1], z=vma_line_original[:, 2],
        mode='lines+markers',
        line=dict(color='orange', width=2),
        marker=dict(size=2, color='orange'),
        name='VMA原始线',
        hovertemplate='VMA原始<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # 5. VMA重采样线（32点）
    fig.add_trace(go.Scatter3d(
        x=vma_line_resampled[:, 0], y=vma_line_resampled[:, 1], z=vma_line_resampled[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=3, color='red'),
        name='VMA重采样(32点)',
        hovertemplate='VMA重采样<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # 6. 中心点
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers',
        marker=dict(size=6, color='yellow', symbol='diamond'),
        name='中心点',
        hovertemplate='中心<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # 7. Refined最终结果
    fig.add_trace(go.Scatter3d(
        x=refined_line_final[:, 0], y=refined_line_final[:, 1], z=refined_line_final[:, 2],
        mode='lines+markers',
        line=dict(color='lime', width=3),
        marker=dict(size=3, color='lime'),
        name='Refined结果',
        hovertemplate='Refined<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # 8. Bounding box可视化
    bbox_min = context_steps['bbox_min']
    bbox_max = context_steps['bbox_max']
    # 绘制bbox的12条边
    bbox_edges = [
        # 底面
        [[bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]]],
        [[bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_min[2]]],
        [[bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_min[2]]],
        [[bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_min[1], bbox_min[2]]],
        # 顶面
        [[bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_min[1], bbox_max[2]]],
        [[bbox_max[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]],
        [[bbox_max[0], bbox_max[1], bbox_max[2]], [bbox_min[0], bbox_max[1], bbox_max[2]]],
        [[bbox_min[0], bbox_max[1], bbox_max[2]], [bbox_min[0], bbox_min[1], bbox_max[2]]],
        # 竖边
        [[bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_min[0], bbox_min[1], bbox_max[2]]],
        [[bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_max[2]]],
        [[bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]],
        [[bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_max[2]]],
    ]
    for edge in bbox_edges:
        edge = np.array(edge)
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'Sample {sample_idx} - VMA采样和Refine推理过程',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig

def main():
    print("="*60)
    print("VMA采样和Refine推理过程可视化")
    print("="*60)

    # 1. 加载数据
    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations...")
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

    # 5. 处理并可视化
    print(f"\n5. 处理并生成采样过程可视化（前{NUM_SAMPLES_TO_VIS}个样本）...")

    annot_timestamps = sorted([int(ts) for ts in annotations.keys()])
    vis_count = 0

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

            print(f"\n处理样本 {vis_count}...")

            # Step 1: 转换到世界坐标
            points_2d = det['points_2d']
            line_world_2d = image_to_world(points_2d, M_cropped_reverse, M_bev2world)
            line_world = np.column_stack([line_world_2d, np.zeros(len(line_world_2d))])
            print(f"  VMA原始线: {len(line_world)} 个点")

            # Step 2: 重采样到32点
            vma_line_resampled = resample_line(line_world, NUM_LINE_POINTS)
            print(f"  重采样后: {len(vma_line_resampled)} 个点")

            # Step 3: 提取context（带步骤信息）
            context_steps = extract_context_with_steps(pcd_points_with_intensity, line_world, CONTEXT_RADIUS)
            print(f"  BBox过滤: {len(context_steps['bbox_points'])} 个点")
            print(f"  Radius过滤: {len(context_steps['radius_points'])} 个点")

            # Step 4: 加权采样
            sampled_context, sampled_weights, all_weights, sampled_indices = weighted_sampling_with_weights(
                context_steps['radius_points'], line_world, NUM_CONTEXT_POINTS, DECAY_SCALE
            )
            print(f"  加权采样: {NUM_CONTEXT_POINTS} 个点")

            # Step 5: 中心化
            center = vma_line_resampled.mean(axis=0)
            context_xyz = sampled_context[:, :3] - center
            context_intensity = sampled_context[:, 3:4]
            context_normalized = np.hstack([context_xyz, context_intensity])
            vma_line_centered = vma_line_resampled - center
            print(f"  中心化: center = [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

            # Step 6: Refine推理
            context_tensor = torch.from_numpy(context_normalized).float().unsqueeze(0).to(device)
            line_tensor = torch.from_numpy(vma_line_centered).float().unsqueeze(0).to(device)

            with torch.no_grad():
                pred_offsets_stack = model(context_tensor, line_tensor)
                pred_offset = pred_offsets_stack[-1, 0].cpu().numpy()

            refined_line_centered = vma_line_centered + pred_offset
            refined_line_final = refined_line_centered + center
            print(f"  推理完成: offset范围 [{pred_offset.min():.3f}, {pred_offset.max():.3f}]")

            # 生成可视化
            fig = create_visualization(
                vis_count, line_world, vma_line_resampled,
                context_steps, sampled_context, sampled_weights,
                vma_line_centered, refined_line_centered, refined_line_final,
                center
            )

            output_path = os.path.join(OUTPUT_DIR, f"sampling_process_{vis_count:03d}.html")
            fig.write_html(output_path)
            print(f"  保存到: {output_path}")

            vis_count += 1

    # 生成索引页面
    index_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VMA采样和Refine推理过程可视化</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .sample-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .sample-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; }}
            .sample-card h3 {{ margin-top: 0; }}
            .sample-card a {{ text-decoration: none; color: #0066cc; }}
            .sample-card a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>VMA采样和Refine推理过程可视化</h1>
        <p>展示VMA检测结果如何采样点云并进行推理的完整过程</p>
        <div class="sample-grid">
    """

    for i in range(vis_count):
        index_html += f"""
            <div class="sample-card">
                <h3>Sample {i}</h3>
                <a href="sampling_process_{i:03d}.html" target="_blank">查看详细过程</a>
            </div>
        """

    index_html += """
        </div>
    </body>
    </html>
    """

    index_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"\n{'='*60}")
    print(f"完成！生成了 {vis_count} 个采样过程可视化")
    print(f"索引页面: {index_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()