import skimage
print(skimage.__version__)

from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np

# Load AI-generated image
image = io.imread('vscode-remote://ssh-remote%2Bskywalker.ece.ucsb.edu/home/avinash/dataDetection/GenImage/train/ai/000_biggan_00000.png')
gray = color.rgb2gray(image)
gray = (gray * 255).astype(np.uint8)  # Convert to 8-bit grayscale

# Compute co-occurrence matrix
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # horizontal, diagonal, vertical, etc.
glcm = greycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

# Visualize one matrix (e.g., horizontal)
plt.imshow(glcm[:, :, 0, 0], cmap='hot')
plt.title("Co-occurrence Matrix (Horizontal) for AI-generated Image")
plt.colorbar()
plt.show()