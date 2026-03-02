import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        # FIX #5: Remove hardcoded intensity gate scaling
        # Let model learn the optimal gating strategy
        self.intensity_gate = nn.Sequential(
            nn.Conv1d(1, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, N), C = 4 -> [x, y, z, intensity]
        # Separate intensity for gating
        intensity = x[:, 3:4, :]  # (B, 1, N)
        feat1 = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        feat2 = F.relu(self.bn2(self.conv2(feat1)))  # (B, 128, N)
        feat3 = F.relu(self.bn3(self.conv3(feat2)))  # (B, 256, N)
        feat4 = F.relu(self.bn4(self.conv4(feat3)))  # (B, 512, N)
        feat5 = F.relu(self.bn5(self.conv5(feat4)))  # (B, 1024, N)

        # Multi-scale fusion
        multi_scale = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        fused = self.fusion(multi_scale)  # (B, 1024, N)

        # FIX #5 + V2_FIX #1: Additive gating to avoid gradient vanishing
        gate = self.intensity_gate(intensity)  # (B, 1024, N), range [0, 1]
        fused = fused + fused * gate  # Additive: fused * (1 + gate), range [1, 2]

        # Dual pooling: max + avg
        max_pool = torch.max(fused, 2, keepdim=False)[0]  # (B, 1024)
        avg_pool = torch.mean(fused, 2, keepdim=False)    # (B, 1024)
        global_feat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2048)

        return global_feat, fused

class PositionalEncoding(nn.Module):
    """
    FIX #2: Sinusoidal positional encoding for 3D coordinates
    More stable and generalizable than MLP-based encoding
    """
    def __init__(self, in_dim=3, out_dim=256, temperature=10000):
        super().__init__()
        self.out_dim = out_dim
        self.temperature = temperature
        self.in_dim = in_dim

    def forward(self, xyz):
        """
        xyz: (B, N, 3) - 3D coordinates
        Returns: (B, N, out_dim) - Positional encoding
        """
        # V2_FIX #2: Normalize each dimension independently to preserve scale
        # This ensures x, y, z all get normalized to [-1, 1] independently
        xyz_max = xyz.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 3)
        xyz_normalized = xyz / xyz_max

        # Sinusoidal encoding - each dimension gets equal features
        num_pos_feats = self.out_dim // (self.in_dim * 2)  # sin + cos for each dim
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = xyz_normalized[..., 0:1] / dim_t
        pos_y = xyz_normalized[..., 1:2] / dim_t
        pos_z = xyz_normalized[..., 2:3] / dim_t

        # Interleave sin and cos
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_z = torch.stack([pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)  # (B, N, num_pos_feats*6)

        # Pad or project to exact out_dim if needed
        if pos.shape[-1] < self.out_dim:
            padding = torch.zeros(pos.shape[0], pos.shape[1], self.out_dim - pos.shape[-1], device=pos.device)
            pos = torch.cat([pos, padding], dim=-1)
        elif pos.shape[-1] > self.out_dim:
            pos = pos[..., :self.out_dim]

        return pos

class DetrTransformerDecoderLayer(nn.Module):
    """
    Standard DETR-style Transformer Decoder Layer.
    Decouples Content (tgt/memory) and Position (query_pos/pos).
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None, pos=None):
        """
        tgt: (B, M, C) - Query Features (Line points)
        memory: (B, N, C) - Key/Value Features (Context)
        query_pos: (B, M, C) - Query Positional Encoding
        pos: (B, N, C) - Memory Positional Encoding
        """
        # 1. Self Attention (Query-Query)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross Attention (Query-Memory)
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        tgt2 = self.cross_attn(q, k, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class LineRefineNet(nn.Module):
    def __init__(self, num_line_points=32, feature_dim=1024):
        super(LineRefineNet, self).__init__()

        # Config
        self.d_model = 256
        self.num_decoder_layers = 6

        # 1. Context Encoder (PointNet)
        self.context_encoder = MultiScalePointNetEncoder(in_channel=4, out_dim=feature_dim)
        self.context_proj = nn.Linear(feature_dim, self.d_model)

        # FIX #1: Add global feature projection
        self.global_proj = nn.Linear(2048, self.d_model)

        # V2_FIX #3: Learnable positional encoding for global token
        self.global_pos_emb = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # FIX #3: Line Encoder with final activation and dropout
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, self.d_model, 1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()  # Add final activation
        )

        # 3. Positional Encoding (Fixed with sinusoidal)
        self.pos_emb = PositionalEncoding(in_dim=3, out_dim=self.d_model)

        # 4. Decoder Layers (Iterative)
        self.decoder_layers = nn.ModuleList([
            DetrTransformerDecoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=1024, dropout=0.1)
            for _ in range(self.num_decoder_layers)
        ])

        # FIX #7: Deeper regression heads with dropout
        self.reg_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 3)  # dx, dy, dz
            )
            for _ in range(self.num_decoder_layers)
        ])

    def forward(self, context, noisy_line):
        """
        context: (B, N, 4) - Point Cloud [x,y,z,i]
        noisy_line: (B, M, 3) - Noisy Polyline [x,y,z]
        Returns:
           all_offsets_stack: (num_layers, B, M, 3) - Predicted offsets at each layer
        """
        B, N, C = context.shape
        M = noisy_line.shape[1]

        # --- 1. Encode Context (Memory) ---
        ctx_trans = context.transpose(2, 1)
        global_feat, ctx_pointwise = self.context_encoder(ctx_trans)

        # FIX #1: Use global feature as additional context token
        memory = self.context_proj(ctx_pointwise.transpose(2, 1))  # (B, N, 256)
        global_token = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, 256)
        memory = torch.cat([global_token, memory], dim=1)  # (B, N+1, 256)

        # Memory Position Embedding
        # V2_FIX #3: Use learnable position for global token
        pos_mem_points = self.pos_emb(context[:, :, :3])  # (B, N, 256)
        pos_mem_global = self.global_pos_emb.expand(B, -1, -1)  # (B, 1, 256)
        pos_mem = torch.cat([pos_mem_global, pos_mem_points], dim=1)  # (B, N+1, 256)

        # --- 2. Encode Line (Initial Query) ---
        line_feat = self.point_mlp(noisy_line.transpose(2, 1))
        tgt = line_feat.transpose(2, 1)  # (B, M, 256)

        # FIX #4: Each layer predicts offset independently (no gradient accumulation)
        all_pred_offsets = []
        current_line_coords = noisy_line.clone()  # For positional encoding only

        # --- 3. Iterative Refinement Loop ---
        for i, (decoder_layer, reg_branch) in enumerate(zip(self.decoder_layers, self.reg_branches)):

            # Dynamic Positional Encoding based on current refined coordinates
            pos_tgt = self.pos_emb(current_line_coords)  # (B, M, 256)

            # Transformer Decoder Layer
            tgt = decoder_layer(tgt, memory, query_pos=pos_tgt, pos=pos_mem)

            # FIX #4: Each layer predicts offset directly relative to noisy_line
            # This avoids gradient accumulation through layers
            pred_offset = reg_branch(tgt)  # (B, M, 3)
            all_pred_offsets.append(pred_offset)

            # Update coordinates for next layer's positional encoding
            # Use detach to prevent gradient flow through coordinate updates
            current_line_coords = noisy_line + pred_offset.detach()

        # Stack outputs: (L, B, M, 3)
        return torch.stack(all_pred_offsets)

if __name__ == '__main__':
    # Test
    fake_ctx = torch.randn(2, 1024, 4)
    fake_line = torch.randn(2, 32, 3)
    model = LineRefineNet()
    out = model(fake_ctx, fake_line)
    print("Output shape:", out.shape)  # Expect (6, 2, 32, 3)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
