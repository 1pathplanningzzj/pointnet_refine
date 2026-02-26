#!/usr/bin/env python3
"""
批量查看所有可视化样本
"""
import os
import glob
import open3d as o3d

VIS_DIR = "/homes/zhangzijian/pointnet_refine/visualizations/vma_refine_vis"

ply_files = sorted(glob.glob(os.path.join(VIS_DIR, "*.ply")))

print(f"找到 {len(ply_files)} 个PLY文件")
print("\n按任意键查看下一个样本，按Q退出\n")

for i, ply_path in enumerate(ply_files):
    txt_path = ply_path.replace('.ply', '.txt')

    print(f"\n{'='*60}")
    print(f"样本 {i+1}/{len(ply_files)}: {os.path.basename(ply_path)}")
    print(f"{'='*60}")

    # 显示文本信息
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            print(f.read())

    # 加载并显示
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"\n正在显示... (关闭窗口查看下一个)")

    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"Sample {i+1}/{len(ply_files)} - {os.path.basename(ply_path)}",
                                      width=1280,
                                      height=720)

print("\n查看完成！")