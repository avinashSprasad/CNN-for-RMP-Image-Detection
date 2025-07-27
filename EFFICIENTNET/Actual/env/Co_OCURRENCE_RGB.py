import torch
import torch.nn as nn

class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, x):
        if x.max() <= 1:
            x = x * 255.0
        x = x.round().long()  # [B, 3, H, W]

        B, C, H, W = x.shape
        device = x.device
        cooc_matrices = []

        directions = [
            (slice(0, H),     slice(0, W - 1), slice(0, H),     slice(1, W)),     # →
            (slice(0, H - 1), slice(0, W),     slice(1, H),     slice(0, W)),     # ↓
            (slice(0, H - 1), slice(0, W - 1), slice(1, H),     slice(1, W)),     # ↘
            (slice(1, H),     slice(0, W - 1), slice(0, H - 1), slice(1, W))      # ↙
        ]

        for ref_H, ref_W, neigh_H, neigh_W in directions:
            ref = x[:, :, ref_H, ref_W]         # [B, 3, H', W']
            neigh = x[:, :, neigh_H, neigh_W]   # [B, 3, H', W']
            N = ref.shape[2] * ref.shape[3]

            ref_flat = ref.reshape(B, C, N)
            neigh_flat = neigh.reshape(B, C, N)

            for ch in range(C):  # Now loops over R, G, B
                co_mat = torch.zeros(B, self.num_levels, self.num_levels, device=device)

                for b in range(B):
                    r = ref_flat[b, ch].clamp(0, self.num_levels - 1)
                    n = neigh_flat[b, ch].clamp(0, self.num_levels - 1)

                    indices = (r * self.num_levels + n).view(-1).long()
                    counts = torch.bincount(indices, minlength=self.num_levels**2).float().to(device)
                    co_matrix = counts.view(self.num_levels, self.num_levels)

                    total = co_matrix.sum()
                    if total > 0:
                        co_matrix = co_matrix / total

                    co_mat[b] = co_matrix

                cooc_matrices.append(co_mat)

        return torch.stack(cooc_matrices, dim=1)  # [B, 12, L, L]
