#!/usr/bin/env python3
"""
简单的PLY文件查看器
用法: python view_ply.py <ply_file_path>
"""
import sys
import open3d as o3d

if len(sys.argv) < 2:
    print("用法: python view_ply.py <ply_file_path>")
    sys.exit(1)

ply_path = sys.argv[1]
print(f"加载PLY文件: {ply_path}")

pcd = o3d.io.read_point_cloud(ply_path)
print(f"点数: {len(pcd.points)}")
print(f"有颜色: {pcd.has_colors()}")

# 可视化
print("\n控制说明:")
print("  - 鼠标左键拖动: 旋转")
print("  - 鼠标右键拖动: 平移")
print("  - 滚轮: 缩放")
print("  - Q: 退出")

o3d.visualization.draw_geometries([pcd],
                                  window_name="PLY Viewer",
                                  width=1280,
                                  height=720,
                                  left=50,
                                  top=50)