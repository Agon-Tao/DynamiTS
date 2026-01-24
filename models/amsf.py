import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicScaleSelector(nn.Module):
    def __init__(self, in_channels, num_scales, reduction=4):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.scale_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                          # [B, C, 1]
            nn.Conv1d(in_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, num_scales, 1),
            nn.Softmax(dim=1)                                # → [B, num_scales, 1]
        )
    def forward(self, x):
        return self.scale_proj(x)  # [B, num_scales, 1]


class AMSF(nn.Module):
    def __init__(self, in_channels, k, c, proj_dim=64, temperature=0.07):
        super().__init__()
        num_scales = k+1
        scale_factors=[c ** i for i in range( 0, k+1)]
        self.selector = DynamicScaleSelector(in_channels, num_scales)
        self.scale_factors = scale_factors
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.weights_log = []

    def forward(self, x):
        """
        Input: x: [B, C, L]
        Return:
          - output: [B, C, L]
        """
        B, C, L = x.shape
        weights = self.selector(x)  # [B, num_scales, 1]
        self.weights_log.append(weights.detach().cpu().numpy())
        outputs, projections = [], []
        for i, s in enumerate(self.scale_factors):
            down = F.avg_pool1d(x, kernel_size=s, stride=s, ceil_mode=True)  # [B, C, L//s]
            feat = self.conv(down)
            up = F.interpolate(feat, size=L, mode='linear', align_corners=True)
            outputs.append(up)
        output_stack = torch.stack(outputs, dim=1)  # [B, S, C, L]
        weights_exp = weights.view(B, len(outputs), 1, 1)  # [B, S, 1, 1]
        fused = torch.sum(weights_exp * output_stack, dim=1)  # [B, C, L]
        return fused, weights