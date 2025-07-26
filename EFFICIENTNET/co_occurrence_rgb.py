import torch
import torch.nn as nn
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class RGBGLCM(nn.Module):
    def __init__(self, num_levels=256, distances=[1], angles=[0, np.pi/2]):
        super().__init__()
        self.num_levels = num_levels
        self.distances = distances
        self.angles = angles  # 2 directions here

    def forward(self, x):
        # x shape: [B, 3, H, W], values in [0, 1]
        B, C, H, W = x.shape
        device = x.device
        outputs = []

        for i in range(B):
            img = x[i].cpu().numpy()  # [3, H, W]
            co_features = []

            for ch in range(3):  # R, G, B channels
                channel = (img[ch] * 255).astype(np.uint8)

                glcm = graycomatrix(
                    channel,
                    distances=self.distances,
                    angles=self.angles,
                    levels=self.num_levels,
                    symmetric=True,
                    normed=True
                )  # shape: [levels, levels, num_dist, num_angles]

                # For each angle separately (here 2), get the matrix [levels, levels]
                # This gives 2 matrices per channel, total 6 for RGB x 2 angles
                for angle_idx in range(len(self.angles)):
                    mat = glcm[:, :, 0, angle_idx]  # levels x levels matrix for distance=1, angle=angle_idx
                    co_features.append(torch.tensor(mat, dtype=torch.float32))

            # co_features list length = 3 channels * 2 angles = 6
            sample_tensor = torch.stack(co_features, dim=0)  # [6, levels, levels]
            outputs.append(sample_tensor)

        # Stack batch: [B, 6, levels, levels]
        return torch.stack(outputs).to(device)