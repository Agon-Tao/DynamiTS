import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def plot_uncertainty_with_patches(
    model,
    batch_x,
    save_path,
    sample_idx=0,
    var_idx=0,
    improve_id=0,
    show_series=True
):

    device = next(model.parameters()).device
    model.eval()

    patcher = model.improves[improve_id]   # type: EnhancedUncertaintyPatch
    x = batch_x.to(device)                             # [B,L,C]
    x_chfirst = x.transpose(1, 2).contiguous()         # [B,C,L]
    _ = patcher(x_chfirst)

    u      = patcher.cached_u[sample_idx, var_idx].detach().cpu().numpy()           # [L]
    lefts  = patcher.cached_left[sample_idx, var_idx].detach().cpu().numpy()       # [N]
    rights = patcher.cached_right[sample_idx, var_idx].detach().cpu().numpy()      # [N]
    L = u.shape[0]
    t = np.arange(L)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 3.2))
    # (a)
    plt.plot(t, u, label='Uncertainty $U(t)$', lw=2)

    # (b)
    for l, r in zip(lefts, rights):
        l_clip = max(0, min(L-1, l))
        r_clip = max(0, min(L-1, r))
        if r_clip < l_clip:
            l_clip, r_clip = r_clip, l_clip
        plt.axvspan(l_clip, r_clip, color='orange', alpha=0.25, linewidth=0)

    # (c)
    if show_series:
        series = batch_x[sample_idx, :, var_idx].cpu().numpy()  # [L]
        if series.std() > 1e-6:
            s = (series - series.min()) / (series.max() - series.min())
            s = s * (u.max() - u.min()) * 0.9 + u.min() * 1.05
            plt.plot(t, s, color='tab:blue', alpha=0.7, lw=1.2, label='Series (scaled)')

    plt.xlabel('Time')
    plt.ylabel('Strength')
    plt.title(f'Uncertainty vs Patch Boundaries  (sample={sample_idx}, var={var_idx})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=300)
    plt.savefig(save_path + '.pdf', dpi=300)
    plt.close()