import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, in_channel=4, out_dim=512):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x: (B, C, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # (B, Out, N)
        x = torch.max(x, 2, keepdim=False)[0] # Global Max Pooling -> (B, Out)
        return x

class LineRefineNet(nn.Module):
    def __init__(self, num_line_points=32, feature_dim=512):
        super(LineRefineNet, self).__init__()
        
        # 1. Context Encoder
        self.context_encoder = PointNetEncoder(in_channel=4, out_dim=feature_dim)
        
        # 2. Line Point Encoder (encode geometry of each line point relative to center)
        # Input: 3 (xyz)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 3. Fusion & Regressor
        # Concatenate: [GlobalContext(512), PointFeat(128)] -> 640
        self.regressor = nn.Sequential(
            nn.Conv1d(feature_dim + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1) # Predict dx, dy, dz
        )

    def forward(self, context, noisy_line):
        """
        context: (B, N, 4) - Point Cloud
        noisy_line: (B, M, 3) - Noisy Polyline
        """
        B = context.shape[0]
        
        # Transpose for Conv1d: (B, C, N)
        ctx = context.transpose(2, 1) 
        line = noisy_line.transpose(2, 1)
        
        # 1. Encode Context -> Global Feature
        global_feat = self.context_encoder(ctx) # (B, 512)
        
        # 2. Encode Line Points
        point_feat = self.point_mlp(line) # (B, 128, M)
        
        # 3. Fuse
        # Expand global feature to match number of line points
        global_feat_expanded = global_feat.unsqueeze(2).repeat(1, 1, point_feat.shape[2]) # (B, 512, M)
        
        fusion = torch.cat([global_feat_expanded, point_feat], dim=1) # (B, 640, M)
        
        # 4. Regress Offsets
        offsets = self.regressor(fusion) # (B, 3, M)
        
        # Transpose back: (B, M, 3)
        return offsets.transpose(2, 1)

if __name__ == '__main__':
    # Test
    fake_ctx = torch.randn(2, 1024, 4)
    fake_line = torch.randn(2, 32, 3)
    model = LineRefineNet()
    out = model(fake_ctx, fake_line)
    print("Output shape:", out.shape) # Expect (2, 32, 3)
