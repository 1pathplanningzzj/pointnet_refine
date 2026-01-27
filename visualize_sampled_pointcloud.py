import os
import argparse

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3d proj

from src.dataset import LaneRefineDataset


def visualize_sample(
    data_root: str,
    index: int = 0,
    num_line_points: int = 32,
    num_context_points: int = 2048,
    crop_radius: float = 4.0,
    decay_scale: float = 2.0,
    save_path: str = None,
    show: bool = True,
):
    """
    可视化 LaneRefineDataset 中某一个样本采样后的点云情况：
      - context 点云（加权采样 + 裁剪 + 归一化后）
      - noisy line（中心化后）
      - GT line（中心化后）
    这里使用的是和训练时完全一样的 Dataset 流程，直观查看“模型看到的输入”。
    """
    dataset = LaneRefineDataset(
        data_root=data_root,
        num_line_points=num_line_points,
        num_context_points=num_context_points,
        crop_radius=crop_radius,
        decay_scale=decay_scale,
    )

    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} out of range, dataset size = {len(dataset)}")

    sample = dataset[index]

    # Tensor -> numpy
    context = sample["context"].numpy()  # (N, 4)  [x, y, z, intensity]，已中心化
    noisy_line = sample["noisy_line"].numpy()  # (M, 3)，已中心化
    target_offset = sample["target_offset"].numpy()  # (M, 3)

    # 由中心化后的 noisy + offset 还原中心化后的 GT
    gt_line = noisy_line + target_offset  # (M, 3)

    # 拆出 xyz / intensity
    ctx_xyz = context[:, :3]
    ctx_intensity = context[:, 3]

    # 简单归一化一下强度用于着色
    if ctx_intensity.max() > ctx_intensity.min():
        colors = (ctx_intensity - ctx_intensity.min()) / (
            ctx_intensity.max() - ctx_intensity.min()
        )
    else:
        colors = ctx_intensity

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 1) context 点云
    ax.scatter(
        ctx_xyz[:, 0],
        ctx_xyz[:, 1],
        ctx_xyz[:, 2],
        c=colors,
        cmap="viridis",
        s=1,
        alpha=0.6,
        label="context points",
    )

    # 2) noisy line（中心化）
    ax.plot(
        noisy_line[:, 0],
        noisy_line[:, 1],
        noisy_line[:, 2],
        color="red",
        linewidth=2.0,
        label="noisy line (centered)",
    )

    # 3) GT line（中心化）
    ax.plot(
        gt_line[:, 0],
        gt_line[:, 1],
        gt_line[:, 2],
        color="green",
        linewidth=2.0,
        label="gt line (centered)",
    )

    ax.set_xlabel("X (centered)")
    ax.set_ylabel("Y (centered)")
    ax.set_zlabel("Z (centered)")
    ax.set_title(f"Sample #{index} - Context Sampling Visualization")
    ax.legend(loc="best")
    ax.view_init(elev=20, azim=60)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize sampled context point cloud + lines for LaneRefineDataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="train_data",
        help="数据根目录（和训练脚本中的 DATA_ROOT 一致）",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="要可视化的样本索引（0 ~ len(dataset)-1）",
    )
    parser.add_argument(
        "--num_line_points",
        type=int,
        default=32,
        help="折线重采样点数（需与训练一致）",
    )
    parser.add_argument(
        "--num_context_points",
        type=int,
        default=2048,
        help="context 采样点数（需与训练一致）",
    )
    parser.add_argument(
        "--crop_radius",
        type=float,
        default=4.0,
        help="沿线截取 context 点云的半径（需与训练一致）",
    )
    parser.add_argument(
        "--decay_scale",
        type=float,
        default=2.0,
        help="加权采样时的距离衰减尺度（需与训练一致）",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="vis_data/sample_context_vis.png",
        help="可视化结果保存路径",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="只保存图片，不弹出窗口（服务器/无显示环境建议加上）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_sample(
        data_root=args.data_root,
        index=args.index,
        num_line_points=args.num_line_points,
        num_context_points=args.num_context_points,
        crop_radius=args.crop_radius,
        decay_scale=args.decay_scale,
        save_path=args.save_path,
        show=not args.no_show,
    )

