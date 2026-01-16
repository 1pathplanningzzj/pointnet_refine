import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RefineDataset(Dataset):
    def __init__(self, data_list, num_points=512, radius=2.0):
        """
        Args:
            data_list: list of dict, 每个元素包含:
                - 'pred_point': np.array (3,) 预测的关键点中心（通常是线段中点）
                - 'line_direction': np.array (2,) 预测线的方向向量 (dx, dy)，用于旋转对齐
                - 'nearby_points': np.array (N, 3) 原始点云 crop
                - 'gt_offset': float 或 shape=(1,) 标量，表示沿法线方向的真实偏移量（单位: m）
                  正负号约定：，需要在数据构造时保持统一
            num_points: PointNet 输入的固定点数
            radius: 上下文半径（这里只在 data_list 构造时用到，类内部不再使用）
        """
        self.data_list = data_list
        self.num_points = num_points
        self.radius = radius

    def __len__(self):
        return len(self.data_list)

    def normalize_pc(self, points, center, direction):
        """
        核心前处理：中心化 + 旋转对齐

        - 将局部点云平移到以预测 keypoint/线段中点为原点
        - 将线方向 direction 对齐到 X 轴（这样 Y 轴天然就是“法线方向”）
        """
        # 1. Centering (移除绝对坐标)
        points = points - center

        # 2. Rotation Alignment (Canonicalization)
        # 将预测线的方向旋转到 X 轴正方向
        # direction = (dx, dy), angle = arctan2(dy, dx)
        # 我们需要旋转 -angle
        angle = -np.arctan2(direction[1], direction[0])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )

        # (N, 3) @ (3, 3) -> (N, 3)
        # 注意: 如果 points 很多，这步可能比较慢，可考虑提前在离线脚本中完成
        points_rotated = np.dot(points, R.T)

        return points_rotated

    def __getitem__(self, idx):
        item = self.data_list[idx]

        raw_points = item["nearby_points"]  # (N, 3)
        center = item["pred_point"]  # (3,)
        direction = item["line_direction"]  # (2,)
        gt_offset = float(item["gt_offset"])  # 标量 Label

        # --- Preprocessing ---

        # 1. 归一化与对齐 (World -> Local Line Frame)
        norm_points = self.normalize_pc(raw_points, center, direction)

        # 2. 采样/补全 (Sampling/Padding)
        curr_n = norm_points.shape[0]
        if curr_n >= self.num_points:
            # 随机采样
            choice = np.random.choice(curr_n, self.num_points, replace=False)
            norm_points = norm_points[choice, :]
        else:
            # 随机重复补全
            if curr_n == 0:
                # 极端情况：附近没有点，填 0，等价于“啥也没看到”
                norm_points = np.zeros((self.num_points, 3), dtype=np.float32)
            else:
                choice = np.random.choice(curr_n, self.num_points, replace=True)
                norm_points = norm_points[choice, :]

        # 3. 转为 Tensor, PointNet 需要 [Channel, N]
        # Input shape: (3, N)
        points_tensor = torch.from_numpy(norm_points).float().transpose(1, 0)
        # Label shape: (1,) 与 PointNetRefine(output_dim=1) 对齐
        label_tensor = torch.tensor([gt_offset], dtype=torch.float32)

        return points_tensor, label_tensor


def generate_mock_training_data(num_samples=1000, noise_range=0.2, roi_half_width=2.0):
    """
    构造“GT 线 + 法向噪声”的模拟训练数据，用来验证整个 pipeline 是否自洽。

    真正上线时，你可以参考这里的逻辑，把:
      - GT 线 (或 GT 中心线的 keypoints)
      - 周围真实点云
    换进来即可。

    Args:
        num_samples: 采样多少个训练样本
        noise_range: 在法线方向上随机加噪的范围 [-noise_range, +noise_range] (m)
        roi_half_width: ROI 的半宽，控制从 GT 线两侧多宽的范围内采点

    Returns:
        data_list: 可以直接喂给 RefineDataset 的 list[dict]
    """
    data_list = []

    for _ in range(num_samples):
        # 1) 随机生成一条“GT 线段”（在 x-y 平面）
        # 起点终点只用来确定方向，不影响核心逻辑
        p_start = np.array(
            [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                0.0,
            ]
        )
        p_end = p_start + np.array(
            [
                np.random.uniform(5, 15),
                np.random.uniform(-5, 5),
                0.0,
            ]
        )
        mid_point = (p_start + p_end) / 2.0  # 用作 pred_point

        # 线方向 (dx, dy)，归一化后只保留平面分量
        tangent = p_end - p_start
        tangent[2] = 0.0
        tangent_norm = np.linalg.norm(tangent[:2]) + 1e-6
        direction_xy = tangent[:2] / tangent_norm  # (2,)

        # 2) 以“GT 线”为基准，在法线方向上加一个随机 offset，模拟 VMA 预测的“带噪线”
        # 约定: 正方向等于右手法线 (normal = [-dy, dx])
        normal_xy = np.array([-direction_xy[1], direction_xy[0]])
        offset_gt = np.random.uniform(-noise_range, noise_range)  # 真值 offset（标量）

        # "预测线" 的中点（相当于 GT 被平移了 offset_gt）
        pred_mid_point = mid_point + np.array(
            [normal_xy[0] * offset_gt, normal_xy[1] * offset_gt, 0.0]
        )

        # 3) 在 “GT 线” 周围均匀撒点，代表真实点云分布（简化版）
        # 这里构造的是贴着 GT 线的点云，而不是预测线
        num_pc = np.random.randint(200, 600)
        ts = np.random.uniform(-5.0, 5.0, size=(num_pc,))  # 沿着切线方向的范围
        # 基于 mid_point 沿 tangent 延伸
        gt_line_points = mid_point[:2] + np.outer(ts, direction_xy)  # (num_pc, 2)
        # 再在法线方向加一个小扰动，模拟车道宽度内的散布
        lateral_noise = np.random.uniform(
            -roi_half_width, roi_half_width, size=(num_pc,)
        )
        pc_xy = (
            gt_line_points + lateral_noise[:, None] * normal_xy[None, :]
        )  # (num_pc, 2)
        # 高度 z 上加一点随机噪声
        pc_z = np.random.uniform(-0.2, 0.2, size=(num_pc, 1))
        nearby_points = np.concatenate([pc_xy, pc_z], axis=1).astype(np.float32)

        # 4) 组织成一个样本:
        # pred_point: 给网络看的“预测线”中点（有噪声）
        # line_direction: 线方向
        # nearby_points: 真实点云（贴着 GT 线）
        # gt_offset: 希望网络学会回归的 offset（即需要“往回走”的距离）
        sample = {
            "pred_point": pred_mid_point.astype(np.float32),
            "line_direction": direction_xy.astype(np.float32),
            "nearby_points": nearby_points,
            # Label: 真值 offset 的负值 —— 网络预测的就是“需要加回去”的距离
            "gt_offset": -offset_gt,
        }
        data_list.append(sample)

    return data_list


# 简单自测
if __name__ == "__main__":
    mock_data = generate_mock_training_data(10)

    ds = RefineDataset(mock_data)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    for pts, lbl in dl:
        print("Batch points:", pts.shape)  # [B, 3, num_points]
        print("Batch labels:", lbl.shape)  # [B, 1]
        print("Label values (m):", lbl[:4].view(-1).numpy())
        break

