import torch
import torch.nn as nn

class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def rgb_to_ycbcr(self, img):
        """Convert RGB [B,3,H,W] to YCbCr [B,3,H,W]"""
        if img.max() <= 1.0:
            img = img * 255.0
        R, G, B = img[:, 0], img[:, 1], img[:, 2]

        Y  =  0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

        ycbcr = torch.stack([Y, Cb, Cr], dim=1) / 255.0  # Normalize to [0,1]
        return ycbcr

    def forward(self, x):
        x = self.rgb_to_ycbcr(x)  # [B, 3, H, W]
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
            ref = x[:, :, ref_H, ref_W]       # [B, C, H', W']
            neigh = x[:, :, neigh_H, neigh_W] # [B, C, H', W']
            N = ref.shape[2] * ref.shape[3]

            ref_flat = ref.reshape(B, C, N)
            neigh_flat = neigh.reshape(B, C, N)

            for ch in range(C):  # Loop over Y, Cb, Cr
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

        return torch.stack(cooc_matrices, dim=1)  # [B, D=12, L, L]
