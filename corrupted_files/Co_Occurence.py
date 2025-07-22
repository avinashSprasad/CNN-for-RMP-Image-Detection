import torch
import torch.nn as nn

def rgb_to_ycbcr(img):
    """Convert RGB [B,3,H,W] to Y [B,1,H,W]"""
    if img.max() <= 1.0:
        img = img * 255.0
    R, G, B = img[:, 0], img[:, 1], img[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y.unsqueeze(1) / 255.0  # Normalize back to [0,1]

class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, x):
        x = rgb_to_ycbcr(x)  # Use Y channel only

        if x.max() <= 1:
            x = x * 255  # Normalize to 0–255 if input in [0, 1]
        x = x.long()

        B, C, H, W = x.shape
        device = x.device
        cooc_matrices = []

        # Diagonal directions only
        directions = [
            (slice(0, H - 1), slice(0, W - 1), slice(1, H), slice(1, W)),  # ↘
            (slice(1, H), slice(1, W), slice(0, H - 1), slice(0, W - 1))   # ↖
        ]

        for ref_H, ref_W, neigh_H, neigh_W in directions:
            selected_ref = []
            selected_neigh = []

            for b in range(B):
                batch_ref = []
                batch_neigh = []
                for c in range(C):  # Only 1 channel after YCbCr
                    ref_slice_2d = x[b, c, ref_H, ref_W]
                    neigh_slice_2d = x[b, c, neigh_H, neigh_W]
                    batch_ref.append(ref_slice_2d)
                    batch_neigh.append(neigh_slice_2d)
                selected_ref.append(batch_ref)
                selected_neigh.append(batch_neigh)

            # Convert lists to tensors: [B, C, H_new, W_new]
            ref_pixels = torch.stack([torch.stack(batch, dim=0) for batch in selected_ref], dim=0)
            neigh_pixels = torch.stack([torch.stack(batch, dim=0) for batch in selected_neigh], dim=0)

            Bc, Cc, Hc, Wc = ref_pixels.shape
            N = Hc * Wc

            # Flatten: [B, C, N]
            ref_flat = ref_pixels.reshape(B, C, N)
            neigh_flat = neigh_pixels.reshape(B, C, N)

            for ch in range(C):
                co_mat = torch.zeros(B, self.num_levels, self.num_levels, device=device)
                for b in range(B):
                    indices = ref_flat[b, ch] * self.num_levels + neigh_flat[b, ch]
                    counts = torch.bincount(indices, minlength=self.num_levels ** 2)
                    co_mat[b] = counts.reshape(self.num_levels, self.num_levels)
                cooc_matrices.append(co_mat)

        # Final output shape: [B, D, num_levels, num_levels]
        output = torch.stack(cooc_matrices, dim=1).float()
        return output


# Test run
test_img = torch.rand(1, 3, 64, 64)  # Random RGB image
processor = CoOccurenceProcessor(num_levels=256)
out = processor(test_img)
print(out.shape)  # [1, 2, 256, 256]
