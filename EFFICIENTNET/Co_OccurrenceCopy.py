'''
import torch
import torch.nn as nn
class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, x):
        # x: [B, C, H, W], expected 0–1 or 0–255 float
        if x.max() <= 1:
            x = x * 255.0
        x = x.round().clamp(0, 255)

        B, C, H, W = x.shape
        device = x.device
        cooc_matrices = []

        directions = [
            (slice(0, H - 1), slice(0, W - 1), slice(1, H), slice(1, W)),  # ↘
            (slice(1, H), slice(1, W), slice(0, H - 1), slice(0, W - 1))   # ↖
        ]

        for ref_H, ref_W, neigh_H, neigh_W in directions:
            ref = x[:, :, ref_H, ref_W]
            neigh = x[:, :, neigh_H, neigh_W]

            B, C, h, w = ref.shape
            N = h * w
            ref_flat = ref.reshape(B, C, N)
            neigh_flat = neigh.reshape(B, C, N)

            for ch in range(C):
                co_mat = torch.zeros(B, self.num_levels, self.num_levels, device=device)
                for b in range(B):
                    r = ref_flat[b, ch]
                    n = neigh_flat[b, ch]
                    indices = (r69464 * self.num_levels + n).long()

                    counts = torch.bincount(indices, minlength=self.num_levels**2)
                    co_mat[b] = counts.reshape(self.num_levels, self.num_levels)
                cooc_matrices.append(co_mat)

        return torch.stack(cooc_matrices, dim=1).float()  # [B, D, L, L]


'''
'''
import torch
import torch.nn as nn

class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def rgb_to_ycbcr(self, img):
        """Convert RGB [B,3,H,W] to Y [B,1,H,W]"""
        if img.max() <= 1.0:
            img = img * 255.0
        R, G, B = img[:, 0], img[:, 1], img[:, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        return Y.unsqueeze(1) / 255.0  # Keep in [0,1] range

    def forward(self, x):
        x = self.rgb_to_ycbcr(x)
        if x.max() <= 1:
            x = x * 255.0
        x = x.round().long()

        B, C, H, W = x.shape
        device = x.device
        cooc_matrices = []

        directions = [
            (slice(0, H - 1), slice(0, W - 1), slice(1, H), slice(1, W)),  # ↘
            (slice(1, H), slice(1, W), slice(0, H - 1), slice(0, W - 1))   # ↖
        ]

        for ref_H, ref_W, neigh_H, neigh_W in directions:
            ref = x[:, :, ref_H, ref_W]
            neigh = x[:, :, neigh_H, neigh_W]
            N = ref.shape[2] * ref.shape[3]
            ref_flat = ref.reshape(B, C, N)
            neigh_flat = neigh.reshape(B, C, N)

            for ch in range(C):  # Usually 1 channel (Y)
                co_mat = torch.zeros(B, self.num_levels, self.num_levels, device=device)
                for b in range(B):
                    r = ref_flat[b, ch].round().clamp(0, self.num_levels - 1).long()
                    n = neigh_flat[b, ch].round().clamp(0, self.num_levels - 1).long()

                    # Sanity checks

                    # Compute co-occurrence indices safely
                    indices = (r * self.num_levels + n).view(-1)
                    indices = (r * self.num_levels + n).view(-1)
                    indices = indices.cpu().long()
                    assert (indices >= 0).all(), "❌ Still found negative indices"
                    counts = torch.bincount(indices, minlength=self.num_levels**2).to(device)

                    co_mat[b] = counts.reshape(self.num_levels, self.num_levels).to(device)

                cooc_matrices.append(co_mat)

        return torch.stack(cooc_matrices, dim=1).float()  # [B, D, L, L]



print("starting...")

# Load and convert to grayscale (Y channel from YCbCr)
img_path = "/home/avinash/dataDetection/genimage/stableDiffusion/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/ai/666_sdv5_00132.png"
img = Image.open(img_path).convert("YCbCr")
Y, _, _ = img.split()
img_np = np.array(Y)

# Initialize co-occurrence matrix (256x256 for 8-bit images)
co_matrix = np.zeros((256, 256), dtype=np.int32)

# Horizontal direction: pixel(i,j) with pixel(i,j+1)
for i in range(img_np.shape[0]):
    for j in range(img_np.shape[1] - 1):
        ref = img_np[i, j]
        neighbor = img_np[i, j + 1]
        co_matrix[ref, neighbor] += 1

# Output matrix shape and sample
print("✅ Co-occurrence matrix shape:", co_matrix.shape)
print("📊 Top-left 5x5 block:\n", co_matrix[:5, :5])

#cbcr 
'''
import torch
import torch.nn as nn

class CoOccurenceProcessor(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def rgb_to_ycbcr(self, img):
        if img.max() <= 1.0:
            img = img * 255.0

        R, G, B = img[:, 0], img[:, 1], img[:, 2]
        Y  =  0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
        return torch.stack([Y, Cb, Cr], dim=1) / 255.0

    def forward(self, x):
        x = self.rgb_to_ycbcr(x)
        if x.max() <= 1:
            x = x * 255.0
        x = x.round().long()

        B, C, H, W = x.shape
        device = x.device
        cooc_matrices = []

        directions = [
            (slice(0, H),     slice(0, W - 1), slice(0, H),     slice(1, W)),
            (slice(0, H - 1), slice(0, W),     slice(1, H),     slice(0, W)),
            (slice(0, H - 1), slice(0, W - 1), slice(1, H),     slice(1, W)),
            (slice(1, H),     slice(0, W - 1), slice(0, H - 1), slice(1, W))
        ]

        for ref_H, ref_W, neigh_H, neigh_W in directions:
            ref = x[:, :, ref_H, ref_W]
            neigh = x[:, :, neigh_H, neigh_W]
            N = ref.shape[2] * ref.shape[3]

            ref_flat = ref.reshape(B, C, N)
            neigh_flat = neigh.reshape(B, C, N)

            # Loop over channels inside each direction
            for ch in range(C):
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

                # Append one matrix per channel per direction, so total = 4 directions * 3 channels = 12
                cooc_matrices.append(co_mat)

        # Stack all 12 matrices along dim=1
        return torch.stack(cooc_matrices, dim=1)
