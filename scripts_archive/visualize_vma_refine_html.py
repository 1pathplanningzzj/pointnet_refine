#!/usr/bin/env python3
"""
将VMA Refine结果可视化为HTML格式
使用Plotly创建交互式3D可视化
"""

import torch
import json
import numpy as np
import open3d as o3d
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.getcwd())
from src.model import LineRefineNet

# ==================== 配置 ====================
VMA_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/results_fuse175_test316.json"
PCD_PATH = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0_annotation_raw_data/merged.pcd"
ANNOT_DIR = "/homes/zhangzijian/vma-dev/testbag/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag/cropped_data/annots"
GT_JSON = "/homes/zhangzijian/pointnet_refine/data/vma_test_data/TAD_front_vision_2025-08-20-11-52-36_85_5to25_0.bag.json"
MODEL_PATH = "/homes/zhangzijian/pointnet_refine/experiments/refine_transformer_based/refine_model_epoch_35.pth"
OUTPUT_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/vma_refine_html"

# Refine参数
CONTEXT_RADIUS = 4.0
NUM_CONTEXT_POINTS = 2048
NUM_LINE_POINTS = 32
DECAY_SCALE = 2.0

# 可视化参数
NUM_SAMPLES_TO_VIS = 10

# ==================== 工具函数（复用） ====================

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
                'attrs': inst.get('attrs', {}),
                'confidence_level': inst.get('confidence_level', 0),
                'line_length_m': inst.get('line_length_m', 0)
            })
    return detections_by_timestamp

def load_annotations(annot_dir):
    annotations = {}
    for annot_file in os.listdir(annot_dir):
        if annot_file.endswith('.json'):
            timestamp = annot_file.replace('.json', '')
            with open(os.path.join(annot_dir, annot_file), 'r') as f:
                annot_data = json.load(f)
                # Extract GT instances with category info
                gt_instances = []
                for instance in annot_data.get('instances', []):
                    position_w = instance.get('position_w', [])
                    category = instance.get('category', 'unknown')
                    if position_w and len(position_w) >= 2:
                        gt_instances.append({
                            'points': np.array(position_w, dtype=np.float32),
                            'category': category
                        })

                annotations[timestamp] = {
                    'M_bev2world': np.array(annot_data['M_bev2world']),
                    'M_cropped_reverse': np.array(annot_data['M_cropped_reverse']),
                    'gt_instances': gt_instances
                }
    return annotations

def load_gt_lines(gt_json_path):
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

def extract_gt_segment(gt_line, vma_line, max_distance=10.0):
    """提取GT线中与VMA检测最接近的分段部分"""
    # 计算GT线每个点到VMA线的最小距离
    distances = np.min(np.linalg.norm(
        gt_line[:, None, :3] - vma_line[None, :, :3], axis=2
    ), axis=1)

    # 找到距离小于阈值的点
    mask = distances < max_distance
    if not mask.any():
        return None

    # 找到连续的分段
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None

    # 取第一个连续分段
    start_idx = indices[0]
    end_idx = indices[-1] + 1

    # 扩展一点以显示完整的分段
    start_idx = max(0, start_idx - 2)
    end_idx = min(len(gt_line), end_idx + 2)

    return gt_line[start_idx:end_idx]

# ==================== HTML可视化 ====================

def create_html_visualization(sample_idx, vma_line_resampled, refined_line, context_points, gt_instances, output_path, timestamp, det_idx):
    """使用Plotly创建交互式HTML可视化"""

    fig = go.Figure()

    # 1. Context点云 - 使用intensity作为颜色
    # Normalize intensity for better visualization
    intensities = context_points[:, 3]
    # 增强对比度：使用平方根变换
    intensities_normalized = np.sqrt(intensities / intensities.max()) if intensities.max() > 0 else intensities

    fig.add_trace(go.Scatter3d(
        x=context_points[:, 0],
        y=context_points[:, 1],
        z=context_points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=intensities_normalized,
            colorscale='Viridis',  # 使用Viridis色图
            cmin=0,
            cmax=1,
            opacity=0.6,
            colorbar=dict(title="Intensity", x=1.15)
        ),
        name='Context Points',
        hovertemplate='Context<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Intensity: %{marker.color:.2f}'
    ))

    # 2. VMA检测线 (红色)
    fig.add_trace(go.Scatter3d(
        x=vma_line_resampled[:, 0],
        y=vma_line_resampled[:, 1],
        z=vma_line_resampled[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(size=5, color='red'),
        name='VMA Detection',
        hovertemplate='VMA<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
    ))

    # 3. Refined线 (绿色)
    fig.add_trace(go.Scatter3d(
        x=refined_line[:, 0],
        y=refined_line[:, 1],
        z=refined_line[:, 2],
        mode='lines+markers',
        line=dict(color='lime', width=8),
        marker=dict(size=5, color='lime'),
        name='Refined',
        hovertemplate='Refined<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
    ))

    # 4. GT线段 - 区分lane_line和curb
    gt_count = 0
    for i, gt_inst in enumerate(gt_instances):
        gt_line = gt_inst['points']
        category = gt_inst['category']

        # 提取与VMA检测最接近的分段
        gt_segment = extract_gt_segment(gt_line, vma_line_resampled, max_distance=10.0)

        if gt_segment is not None and len(gt_segment) >= 2:
            # 根据category选择颜色
            if category == 'lane_line':
                color = 'cyan'
                name = f'GT Lane {gt_count}'
            elif category == 'curb':
                color = 'orange'
                name = f'GT Curb {gt_count}'
            else:
                color = 'magenta'
                name = f'GT {category} {gt_count}'

            fig.add_trace(go.Scatter3d(
                x=gt_segment[:, 0],
                y=gt_segment[:, 1],
                z=gt_segment[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=6),
                marker=dict(size=4, color=color),
                name=name,
                hovertemplate=f'{name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}'
            ))
            gt_count += 1

    # 计算offset
    offset_magnitude = np.linalg.norm(refined_line - vma_line_resampled, axis=1).mean()

    # 布局设置
    fig.update_layout(
        title=dict(
            text=f'Sample {sample_idx} - Timestamp: {timestamp} - Detection: {det_idx}<br>' +
                 f'Average Offset: {offset_magnitude:.4f}m | Intensity Range: [{intensities.min():.0f}, {intensities.max():.0f}]',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='closest'
    )

    # 保存HTML
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)

    print(f"保存HTML到: {html_path} (offset: {offset_magnitude:.4f}m, intensity: {intensities.min():.0f}-{intensities.max():.0f})")

    return offset_magnitude

# ==================== 主流程 ====================

def main():
    print("="*60)
    print("VMA Refine结果HTML可视化")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("\n1. 加载VMA检测结果...")
    vma_detections = load_vma_results(VMA_JSON)

    print("\n2. 加载annotations (包含GT线)...")
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
    print("\n5. 加载refine模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = LineRefineNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 5. 处理并生成HTML可视化
    print(f"\n5. 处理并生成HTML可视化（前{NUM_SAMPLES_TO_VIS}个样本）...")

    annot_timestamps = sorted([int(ts) for ts in annotations.keys()])
    vis_count = 0
    all_offsets = []

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

            # 生成HTML可视化
            output_path = os.path.join(OUTPUT_DIR, f"sample_{vis_count:03d}_ts{timestamp}_det{det_idx}.png")
            offset = create_html_visualization(vis_count, line_resampled, refined_line, context, annot['gt_instances'], output_path, timestamp, det_idx)
            all_offsets.append(offset)

            vis_count += 1
            print(f"已生成 {vis_count}/{NUM_SAMPLES_TO_VIS} 个HTML文件")

    # 创建索引页面
    create_index_page(OUTPUT_DIR, vis_count, all_offsets)

    print(f"\n完成！HTML可视化结果保存在: {OUTPUT_DIR}")
    print(f"打开 {OUTPUT_DIR}/index.html 查看所有样本")

def create_index_page(output_dir, num_samples, offsets):
    """创建索引页面"""
    html_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.html') and f != 'index.html'])

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VMA Refine Visualization Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .card {{
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .card a {{
            text-decoration: none;
            color: #333;
        }}
        .card h3 {{
            margin-top: 0;
            color: #2196F3;
        }}
        .offset {{
            color: #666;
            font-size: 14px;
        }}
        .legend {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 15px;
        }}
        .color-box {{
            display: inline-block;
            width: 20px;
            height: 12px;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <h1>VMA Refine Visualization Results</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Samples:</strong> {num_samples}</p>
        <p><strong>Average Offset:</strong> {np.mean(offsets):.4f} m</p>
        <p><strong>Median Offset:</strong> {np.median(offsets):.4f} m</p>
        <p><strong>Min Offset:</strong> {np.min(offsets):.4f} m</p>
        <p><strong>Max Offset:</strong> {np.max(offsets):.4f} m</p>

        <div class="legend">
            <strong>Legend:</strong><br>
            <span class="legend-item">
                <span class="color-box" style="background-color: gray;"></span>
                Context Points (2048 points, 4m radius)
            </span>
            <span class="legend-item">
                <span class="color-box" style="background-color: red;"></span>
                VMA Detection (resampled to 32 points)
            </span>
            <span class="legend-item">
                <span class="color-box" style="background-color: green;"></span>
                Refined Line (32 points)
            </span>
            <span class="legend-item">
                <span class="color-box" style="background-color: blue;"></span>
                GT Line (if nearby)
            </span>
        </div>
    </div>

    <h2>Samples</h2>
    <div class="grid">
"""

    for i, (html_file, offset) in enumerate(zip(html_files, offsets)):
        html_content += f"""
        <div class="card">
            <a href="{html_file}">
                <h3>Sample {i}</h3>
                <p class="offset">Offset: {offset:.4f} m</p>
                <p style="font-size: 12px; color: #999;">{html_file}</p>
            </a>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"创建索引页面: {index_path}")

if __name__ == "__main__":
    main()