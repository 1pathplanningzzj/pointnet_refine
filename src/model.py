import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetRefine(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        """
        Args:
            input_dim: 输入点特征维度 (x,y,z) 为 3，如有 intensity 则为 4
            output_dim: 回归偏移量维度. 
                        通常为 1 (只回归法向距离 lateral offset)，
                        或者 2 (回归 dx, dy)
        """
        super(PointNetRefine, self).__init__()
        
        # 1. Point-wise Feature Extraction (Shared MLP)
        # 输入: [Batch, Input_Dim, Num_Points]
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # 2. Regression Head (MLP)
        # 输入: [Batch, 1024] (Global Feature)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim) 

    def forward(self, x):
        """
        x: [B, D, N]  (Batch, Dim, Num_Points)
        """
        B, D, N = x.size()

        # --- Feature Extraction ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
        # x shape: [B, 1024, N]

        # --- Max Pooling (Symmetric Function) ---
        # 提取全局几何特征
        x = torch.max(x, 2, keepdim=False)[0] 
        # x shape: [B, 1024]

        # --- Regression Head ---
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # 最后一层通常不加激活函数，因为offset可正可负
        offset = self.fc3(x) 
        
        return offset

if __name__ == '__main__':
    # Test shape
    sim_data = torch.rand(32, 3, 512) # batch=32, xyz=3, points=512
    model = PointNetRefine(input_dim=3, output_dim=1)
    output = model(sim_data)
    print("Input shape:", sim_data.shape)
    print("Output offset shape:", output.shape) # Should be [32, 1]
