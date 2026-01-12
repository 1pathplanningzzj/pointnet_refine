import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RefineDataset(Dataset):
    def __init__(self, data_list, num_points=512, radius=2.0):
        """
        Args:
            data_list: list of dict, 每个元素包含:
                - 'pred_point': np.array (x, y, z) 预测的关键点中心
                - 'line_direction': np.array (dx, dy) 预测线的方向向量（用于旋转对齐）
                - 'nearby_points': np.array (N, 3) 原始点云crop
                - 'gt_offset': np.array (dx, dy) 真实的偏移量标签
            num_points: PointNet 输入的固定点数
            radius: 上下文半径
        """
        self.data_list = data_list
        self.num_points = num_points
        self.radius = radius

    def __len__(self):
        return len(self.data_list)
    
    def normalize_pc(self, points, center, direction):
        """
        核心前处理：中心化 + 旋转对齐
        """
        # 1. Centering (移除绝对坐标)
        points = points - center
        
        # 2. Rotation Alignment (Canonicalization)
        # 将预测线的方向旋转到 X 轴正方向
        # direction = (dx, dy), angle = arctan2(dy, dx)
        # 我们需要旋转 -angle
        angle = -np.arctan2(direction[1], direction[0])
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0],
                      [s, c,  0],
                      [0, 0,  1]])
        
        # Transpose for matmul: (N, 3) @ (3, 3) -> (N, 3)
        # 注意: 如果 points 很多，这步可能比较慢，放在 C++ 或预处理里做更好
        points_rotated = np.dot(points, R.T)
        
        return points_rotated

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        raw_points = item['nearby_points']
        center = item['pred_point']
        direction = item['line_direction']
        gt_offset = item['gt_offset'] # Label

        # --- Preprocessing ---
        
        # 1. 归一化与对齐
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
                # 极端情况：附近没有点，填0
                norm_points = np.zeros((self.num_points, 3))
            else:
                choice = np.random.choice(curr_n, self.num_points, replace=True)
                norm_points = norm_points[choice, :]
        
        # 3. 转为 Tensor, PointNet 需要 [Channel, N]
        # Input shape: (3, N)
        points_tensor = torch.from_numpy(norm_points).float().transpose(1, 0)
        label_tensor = torch.from_numpy(gt_offset).float()

        return points_tensor, label_tensor

# 模拟数据调试
if __name__ == "__main__":
    # Mock data
    mock_data = [{
        'pred_point': np.array([10.0, 10.0, 0.0]),
        'line_direction': np.array([1.0, 1.0]), # 45度方向
        'nearby_points': np.random.rand(100, 3) + np.array([10,10,0]),
        'gt_offset': np.array([0.1, -0.2])
    }] * 10 
    
    ds = RefineDataset(mock_data)
    dl = DataLoader(ds, batch_size=2)
    
    for pts, lbl in dl:
        print("Batch points:", pts.shape) # [2, 3, 512]
        print("Batch labels:", lbl.shape) # [2, 2]
        break
