import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScalePointNetEncoder(nn.Module):
    """Multi-scale PointNet encoder with dual pooling"""
    def __init__(self, in_channel=4, out_dim=1024):
        super(MultiScalePointNetEncoder, self).__init__()
        # Multi-scale feature extraction
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, out_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(out_dim)

        # Fusion layer for multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 128 + 256 + 512 + out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, C, N)
        feat1 = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        feat2 = F.relu(self.bn2(self.conv2(feat1)))  # (B, 128, N)
        feat3 = F.relu(self.bn3(self.conv3(feat2)))  # (B, 256, N)
        feat4 = F.relu(self.bn4(self.conv4(feat3)))  # (B, 512, N)
        feat5 = F.relu(self.bn5(self.conv5(feat4)))  # (B, 1024, N)

        # Multi-scale fusion
        multi_scale = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        fused = self.fusion(multi_scale)  # (B, 1024, N)

        # Dual pooling: max + avg
        max_pool = torch.max(fused, 2, keepdim=False)[0]  # (B, 1024)
        avg_pool = torch.mean(fused, 2, keepdim=False)    # (B, 1024)
        global_feat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2048)

        return global_feat, fused  # Return both global and point-wise features

class CrossAttentionModule(nn.Module):
    """Transformer-style cross-attention: line points attend to context"""
    def __init__(self, embed_dim=256, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key_value):
        """
        query: (B, M, C) - line points
        key_value: (B, N, C) - context points
        """
        # Cross-attention
        attn_out, _ = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query

class ResidualBlock(nn.Module):
    """Residual block for regressor"""
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, 1)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # Shortcut connection
        self.shortcut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class LineRefineNet(nn.Module):
    def __init__(self, num_line_points=32, feature_dim=1024):
        super(LineRefineNet, self).__init__()

        # 1. Multi-scale Context Encoder with dual pooling
        self.context_encoder = MultiScalePointNetEncoder(in_channel=4, out_dim=feature_dim)

        # 2. Line Point Encoder (deeper)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 3. Projection layers for cross-attention
        self.line_proj = nn.Linear(256, 256)
        self.context_proj = nn.Linear(feature_dim, 256)

        # 4. Cross-Attention Module
        self.cross_attention = CrossAttentionModule(embed_dim=256, num_heads=8)

        # 5. Fusion & Deep Regressor with Residual Blocks
        # Input: [GlobalContext(2048), AttendedLineFeat(256)] -> 2304
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(feature_dim * 2 + 256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # Predict dx, dy, dz
        )

    def forward(self, context, noisy_line):
        """
        context: (B, N, 4) - Point Cloud
        noisy_line: (B, M, 3) - Noisy Polyline
        """
        B, N, _ = context.shape
        M = noisy_line.shape[1]

        # Transpose for Conv1d: (B, C, N)
        ctx = context.transpose(2, 1)
        line = noisy_line.transpose(2, 1)

        # 1. Encode Context -> Global Feature + Point-wise Features
        global_feat, ctx_pointwise = self.context_encoder(ctx)  # (B, 2048), (B, 1024, N)

        # 2. Encode Line Points
        point_feat = self.point_mlp(line)  # (B, 256, M)

        # 3. Cross-Attention: line points attend to context
        # Prepare for attention: (B, M, 256) and (B, N, 256)
        line_tokens = self.line_proj(point_feat.transpose(2, 1))  # (B, M, 256)
        ctx_tokens = self.context_proj(ctx_pointwise.transpose(2, 1))  # (B, N, 256)

        attended_line = self.cross_attention(line_tokens, ctx_tokens)  # (B, M, 256)
        attended_line = attended_line.transpose(2, 1)  # (B, 256, M)

        # 4. Fuse: [GlobalContext, AttendedLineFeat]
        global_feat_expanded = global_feat.unsqueeze(2).repeat(1, 1, M)  # (B, 2048, M)
        fusion = torch.cat([global_feat_expanded, attended_line], dim=1)  # (B, 2304, M)

        fusion = self.fusion_proj(fusion)  # (B, 512, M)

        # 5. Regress Offsets
        offsets = self.regressor(fusion)  # (B, 3, M)

        # Transpose back: (B, M, 3)
        return offsets.transpose(2, 1)

if __name__ == '__main__':
    # Test
    fake_ctx = torch.randn(2, 1024, 4)
    fake_line = torch.randn(2, 32, 3)
    model = LineRefineNet()
    out = model(fake_ctx, fake_line)
    print("Output shape:", out.shape)  # Expect (2, 32, 3)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
