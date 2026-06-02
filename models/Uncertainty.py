import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedUncertaintyPatch(nn.Module):
    def __init__(self, input_len, patch_num, patch_len, gamma=1.0, deform_range=0.25, D=16, in_channels=1):
        super().__init__()
        self.input_len = input_len
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.D = D
        self.deform_range = deform_range
        self.in_channels = in_channels

        # === Learnable Offsets ===
        self.offset_predictor = nn.Linear(input_len, patch_num * 3)

        # === Multi-source uncertainty ===
        self.conv_var = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv_resid = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.att_gate = nn.Sequential(
            nn.Conv1d(in_channels, 8 * in_channels, 3, padding=1, groups=in_channels),
            nn.ReLU(),
            nn.Conv1d(8 * in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Sigmoid()
        )
    def compute_uncertainty(self, x):  # [B, C, L]
        var_u = torch.abs(self.conv_var(x))
        resid_u = torch.abs(self.conv_resid(x - x.mean(dim=-1, keepdim=True)))
        attn_u = self.att_gate(x)
        return (var_u + resid_u) * attn_u  # [B, C, L]

    def _get_grid(self, idx, L):  # idx: [B]
        B = idx.shape[0]
        norm_idx = 2 * idx / (L - 1) - 1  # [-1, 1]
        x_grid = norm_idx.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        y_grid = torch.zeros_like(x_grid)
        return torch.cat([x_grid, y_grid], dim=-1)  # [B, 1, 1, 2]

    def interpolate(self, x, left, right):  # x: [B, C, L], left/right: [B, C, patch_num]
        B, C, L = x.shape
        N = self.patch_num
        D = self.D

        t = torch.linspace(0, 1, D, device=x.device).view(1, 1, 1, -1)
        l = left.unsqueeze(-1)
        r = right.unsqueeze(-1)
        pos = l * (1 - t) + r * t  # [B, C, N, D]
        pos = torch.clamp(pos, 0, L - 1)

        x_grid = 2 * pos / (L - 1) - 1  # [-1, 1]
        x_grid = x_grid.reshape(B * C, 1, N * D, 1)
        y_grid = torch.zeros_like(x_grid)
        grid = torch.cat([x_grid, y_grid], dim=-1)  # [B*C, 1, N*D, 2]

        x = x.contiguous().view(B * C, 1, 1, L)  # [B*C, 1, 1, L]

        sampled = F.grid_sample(x, grid, align_corners=True)  # [B*C, 1, 1, N*D]

        patch = sampled.reshape(B, C, N, D)  # ✅ reshape 
        return patch

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        u = self.compute_uncertainty(x)  # [B, C, L]
        anchors = torch.linspace(0, L - 1, self.patch_num).to(x.device).view(1, 1, -1).expand(B, C, -1)  # [B, C, patch_num]

        offset_input = x.mean(dim=1)  # [B, L]
        offset_raw = self.offset_predictor(offset_input).view(B, self.patch_num, 3)  # [B, patch_num, 3]
        delta_c = torch.tanh(offset_raw[:, :, 0]) * (self.patch_len / 2) * self.deform_range
        delta_l = F.relu(torch.tanh(offset_raw[:, :, 1])) * (self.patch_len / 2) * self.deform_range
        delta_r = F.relu(torch.tanh(offset_raw[:, :, 2])) * (self.patch_len / 2) * self.deform_range

        u_avg = F.avg_pool1d(u, kernel_size=self.patch_len, stride=1, padding=self.patch_len // 2)  # [B, C, L]
        gamma_eff = F.softplus(self.gamma)

        u_left_list, u_right_list = [], []
        for i in range(self.patch_num):
            idx_l = anchors[:, 0, i] - self.patch_len / 2  # [B]
            idx_r = anchors[:, 0, i] + self.patch_len / 2  # [B]
            grid_l = self._get_grid(idx_l, L)  # [B, 1, 1, 2]
            grid_r = self._get_grid(idx_r, L)
            input_ = u_avg.unsqueeze(2)  # [B, C, 1, L]
            u_l = F.grid_sample(input_, grid_l, align_corners=True).squeeze(2).squeeze(2)  # [B, C]
            u_r = F.grid_sample(input_, grid_r, align_corners=True).squeeze(2).squeeze(2)
            u_left_list.append(u_l.unsqueeze(2))  # [B, C, 1]
            u_right_list.append(u_r.unsqueeze(2))

        u_left = torch.cat(u_left_list, dim=2)  # [B, C, patch_num]
        u_right = torch.cat(u_right_list, dim=2)

        left = anchors + delta_c.unsqueeze(1) - (self.patch_len / 2 + delta_l.unsqueeze(1) + gamma_eff * u_left)
        right = anchors + delta_c.unsqueeze(1) + (self.patch_len / 2 + delta_r.unsqueeze(1) + gamma_eff * u_right)
        patch = self.interpolate(x, left, right)  # [B, C, patch_num, D]

        self.cached_u = u
        self.cached_left = left
        self.cached_right = right
        return patch  # [B, C, patch_num, D]
