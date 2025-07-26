import torch
import torch.nn as nn
import numpy as np
from skimage.feature import graycomatrix
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

class GrayGLCM(nn.Module):
    def __init__(self, num_levels=256, distances=[1], angles=[0]):
        super().__init__()
        self.num_levels = num_levels
        self.distances = distances
        self.angles = angles

    def forward(self, x):
        B, C, H, W = x.shape

        # Unnormalize from ImageNet stats (to approx [0,1] range)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_unnorm = x * std + mean

        glcm_imgs = []

        for i in range(B):
            img = x_unnorm[i]  # [3, H, W]
            img_np = img.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            img_gray = rgb2gray(img_np)  # float [H, W] in [0,1]
            img_uint8 = img_as_ubyte(img_gray)  # uint8 [H, W]

            glcm = graycomatrix(img_uint8,
                                distances=self.distances,
                                angles=self.angles,
                                levels=self.num_levels,
                                symmetric=True,
                                normed=True)
            # glcm shape: [levels, levels, num_distances, num_angles]
            # Rearrange to [num_distances*num_angles, levels, levels]
            glcm_t = torch.tensor(glcm, dtype=torch.float32).permute(2, 3, 0, 1)
            glcm_t = glcm_t.reshape(len(self.distances)*len(self.angles), self.num_levels, self.num_levels)

            glcm_imgs.append(glcm_t)

        return torch.stack(glcm_imgs).to(x.device)  # [B, D*A, L, L]
