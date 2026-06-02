import torch
import torch.nn as nn
from models.Common import RevIN
from models.Uncertainty import  EnhancedUncertaintyPatch
from models.Dmoe import Dynami
from models.Amsf import AMSF


class DynamiTS(nn.Module):
    """Implementation of Dynami."""
    def __init__(self, input_shape, pred_len, n_block, dropout, patch,k,c, alpha,target_slice, norm=True, channel_wise=False):
        super(DynamiTS, self).__init__()
        # print("✅ Using correct DynamiTS class with channel_wise =", channel_wise)
        self.target_slice = target_slice
        self.norm = norm
        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])
        self.pastmixing = AMSF(in_channels=input_shape[1], k=k, c=c)
        self.input_len = input_shape[0]
        self.in_channels = input_shape[1]
        self.patch_len = patch
        self.patch_num = self.input_len // self.patch_len
        self.D = patch
        self.channel_wise = channel_wise
        self.k=k
        self.c = c
        self.improves = nn.ModuleList([EnhancedUncertaintyPatch(
            input_len=input_shape[0],
            patch_len=patch,
            patch_num=input_shape[0] // patch,
            gamma=1.0,
            deform_range=0.25,
            D=patch,
            in_channels=self.in_channels
        )for _ in range(n_block)])

        self.reconstruct_linear = nn.Linear(self.patch_num * self.D, self.input_len)
        self.moe =Dynami(input_shape, pred_len,  dropout=dropout, num_experts=8, top_k=1)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]
        B, L, C = x.shape
        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]
        time_embedding, weights = self.pastmixing(x)
        for improve in self.improves:
            x = improve(x)
            patch_feat = x.view(B, C, -1)  # [B, C, patch_num * D]
            x = self.reconstruct_linear(patch_feat)  # → [B, C, L]
        x_modify=x
        # MOE
        x, moe_loss = self.moe(x_modify, time_embedding)  # seq_len -> pred_len
        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]
        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]
        if self.target_slice:
            x = x[:, :, self.target_slice]
        return x, moe_loss,weights
