"""
使用 matplotlib 进行 2D 可视化（不依赖 OpenGL，适合 headless 服务器）
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from typing import List, Optional

# 添加 src 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.real_dataset import load_pcd, load_lane_json, RealRefineDataset


def visualize_2d(
    pcd_path: str,
    lane_json_path: Optional[str] = None,
    vma_json_path: Optional[str] = None,
    roi_line_segment: Optional[np.ndarray] = None,
    roi_points: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """
    使用 matplotlib 进行 2D 可视化（俯视图，x-y 平面）
    
    Args:
        pcd_path: PCD 文件路径
        lane_json_path: GT 标注 JSON 路径（可选）
        vma_json_path: VMA 预测 JSON 路径（可选）
        roi_line_segment: (2, 3) ROI 线段（可选）
        roi_points: (M, 3) ROI 内的点云（可选）
        save_path: 保存路径（可选）
        show_plot: 是否显示（在 headless 服务器上设为 False）
    """
    # 1. 加载点云
    print(f"Loading point cloud from {pcd_path}...")
    points = load_pcd(pcd_path)  # (N, 3)
    print(f"  Total points: {len(points)}")
    
    # 2. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # 3. 绘制原始点云（灰色，采样以减少点数）
    if len(points) > 50000:
        # 如果点太多，随机采样
        sample_idx = np.random.choice(len(points), 50000, replace=False)
        points_vis = points[sample_idx]
    else:
        points_vis = points
    
    ax.scatter(
        points_vis[:, 0], points_vis[:, 1],
        c='gray', s=0.1, alpha=0.3, label='Original Point Cloud'
    )
    
    # 4. 绘制 ROI 点（绿色）
    if roi_points is not None and len(roi_points) > 0:
        ax.scatter(
            roi_points[:, 0], roi_points[:, 1],
            c='green', s=2, alpha=0.6, label='ROI Points'
        )
    
    # 5. 绘制 GT 车道线（蓝色）
    if lane_json_path and os.path.exists(lane_json_path):
        print(f"Loading GT lanes from {lane_json_path}...")
        gt_lanes = load_lane_json(lane_json_path)
        print(f"  Found {len(gt_lanes)} GT lanes")
        
        for i, lane in enumerate(gt_lanes):
            ax.plot(
                lane[:, 0], lane[:, 1],
                'b-', linewidth=2, alpha=0.8,
                label='GT Lane' if i == 0 else ''
            )
    
    # 6. 绘制 VMA 预测车道线（红色）
    if vma_json_path and os.path.exists(vma_json_path):
        print(f"Loading VMA predictions from {vma_json_path}...")
        vma_lanes = load_lane_json(vma_json_path)
        print(f"  Found {len(vma_lanes)} VMA lanes")
        
        for i, lane in enumerate(vma_lanes):
            ax.plot(
                lane[:, 0], lane[:, 1],
                'r--', linewidth=2, alpha=0.8,
                label='VMA Prediction' if i == 0 else ''
            )
    
    # 7. 绘制 ROI 线段和边界（黄色）
    if roi_line_segment is not None:
        # ROI 线段
        ax.plot(
            roi_line_segment[:, 0], roi_line_segment[:, 1],
            'y-', linewidth=3, alpha=0.9, label='ROI Segment'
        )
        
        # 计算 ROI 边界
        tangent = roi_line_segment[1] - roi_line_segment[0]
        tangent_norm = np.linalg.norm(tangent[:2]) + 1e-6
        direction_xy = tangent[:2] / tangent_norm
        normal_xy = np.array([-direction_xy[1], direction_xy[0], 0])
        
        roi_half_width = 2.0  # 假设 2 米
        offset_vec = normal_xy * roi_half_width
        
        # 上边界
        upper_line = roi_line_segment + offset_vec
        ax.plot(
            upper_line[:, 0], upper_line[:, 1],
            'y--', linewidth=1, alpha=0.6, label='ROI Boundary'
        )
        
        # 下边界
        lower_line = roi_line_segment - offset_vec
        ax.plot(
            lower_line[:, 0], lower_line[:, 1],
            'y--', linewidth=1, alpha=0.6
        )
        
        # 绘制 ROI 区域（半透明矩形）
        mid_point = (roi_line_segment[0] + roi_line_segment[1]) / 2.0
        length = np.linalg.norm(tangent[:2])
        angle = np.arctan2(tangent[1], tangent[0]) * 180 / np.pi
        
        rect = Rectangle(
            (mid_point[0] - length/2, mid_point[1] - roi_half_width),
            length, 2 * roi_half_width,
            angle=angle, alpha=0.1, facecolor='yellow', edgecolor='yellow'
        )
        ax.add_patch(rect)
    
    # 8. 设置图形属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Point Cloud and Lane Lines Visualization (Top View)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 9. 设置合适的显示范围
    if roi_points is not None and len(roi_points) > 0:
        # 以 ROI 点为中心
        x_min, x_max = roi_points[:, 0].min() - 5, roi_points[:, 0].max() + 5
        y_min, y_max = roi_points[:, 1].min() - 5, roi_points[:, 1].max() + 5
    elif roi_line_segment is not None:
        # 以 ROI 线段为中心
        x_min, x_max = roi_line_segment[:, 0].min() - 10, roi_line_segment[:, 0].max() + 10
        y_min, y_max = roi_line_segment[:, 1].min() - 10, roi_line_segment[:, 1].max() + 10
    else:
        # 使用所有点
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 10. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_dataset_sample_matplotlib(
    dataset: RealRefineDataset,
    idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    使用 matplotlib 可视化 Dataset 中的一个样本
    """
    print(f"\n=== Visualizing Dataset Sample {idx} (2D Top View) ===")
    
    if idx >= len(dataset):
        print(f"Error: Index {idx} out of range (dataset size: {len(dataset)})")
        return
    
    # 加载样本
    points_tensor, label_tensor, meta = dataset[idx]
    
    print(f"Sample info:")
    print(f"  Lidar timestamp: {meta.get('lidar_timestamp', 'N/A')}")
    print(f"  Camera11 timestamp: {meta.get('camera11_timestamp', 'N/A')}")
    print(f"  GT offset: {label_tensor.item():.4f} m")
    print(f"  Original points: {len(meta.get('original_points', []))}")
    print(f"  ROI points: {len(meta.get('nearby_points', []))}")
    
    # 获取路径
    sample = dataset.samples[idx]
    pcd_path = sample['lidar_path']
    gt_path = sample.get('gt_label_path')
    vma_path = sample.get('vma_pred_path')
    
    # 可视化
    visualize_2d(
        pcd_path=pcd_path,
        lane_json_path=gt_path,
        vma_json_path=vma_path,
        roi_line_segment=meta.get('pred_line_segment'),
        roi_points=meta.get('nearby_points'),
        save_path=save_path,
        show_plot=False,  # headless 服务器上设为 False
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud and lane lines (2D matplotlib)")
    parser.add_argument(
        '--data_root',
        type=str,
        default="/homes/zhangzijian/pointnet_refine/data/test/TAD_front_lidar_2025-12-01-13-32-45_25_all.bag",
        help="Data root directory",
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'dataset'],
        default='dataset',
        help="Visualization mode",
    )
    parser.add_argument(
        '--pcd_path',
        type=str,
        default=None,
        help="PCD file path (for single mode)",
    )
    parser.add_argument(
        '--gt_json',
        type=str,
        default=None,
        help="GT lane JSON path (for single mode)",
    )
    parser.add_argument(
        '--vma_json',
        type=str,
        default=None,
        help="VMA prediction JSON path (for single mode)",
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help="Sample index (for dataset mode)",
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='vis_2d.png',
        help="Path to save visualization image",
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.pcd_path:
            print("Error: --pcd_path is required for single mode")
            return
        
        visualize_2d(
            pcd_path=args.pcd_path,
            lane_json_path=args.gt_json,
            vma_json_path=args.vma_json,
            save_path=args.save_path,
            show_plot=False,
        )
    else:
        dataset = RealRefineDataset(
            data_root=args.data_root,
            num_points=512,
            roi_half_width=2.0,
            roi_length=5.0,
            noise_range=0.2,
            use_gt_for_training=True,
            split='train',
        )
        
        visualize_dataset_sample_matplotlib(
            dataset=dataset,
            idx=args.sample_idx,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    main()
