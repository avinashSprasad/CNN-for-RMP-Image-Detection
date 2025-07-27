import numpy as np
from PIL import Image
import matplotlib
print("Matplotlib backend:", matplotlib.get_backend())  # Prints current backend

# Optionally, force a GUI backend if you know one is installed:
# matplotlib.use('TkAgg')  # Uncomment if needed before importing pyplot

import matplotlib.pyplot as plt
from skimage.color import rgb2ycbcr
import matplotlib.patches as patches
import seaborn as sns

# --- Load image from your provided path ---
image_path = "/home/avinash/dataDetection/GenImage/train/nature/n04389033_4250.JPEG"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# --- Convert to YCbCr and extract Y channel ---
ycbcr = rgb2ycbcr(image_np)
y_channel = ycbcr[:, :, 0]

# --- Patch location and size ---
row, col = 0, 0  # Change this to pick a different patch
patch_size = 10
patch_y = y_channel[row:row+patch_size, col:col+patch_size]

# --- Normalize Y values to range 1–10 ---
y_min, y_max = patch_y.min(), patch_y.max()
normalized_patch = 1 + 9 * (patch_y - y_min) / (y_max - y_min)
normalized_patch = normalized_patch.astype(np.uint8)

# --- Compute horizontal co-occurrence matrix (0°) ---
def compute_cooccurrence_matrix(patch, levels=11):
    co_matrix = np.zeros((levels, levels), dtype=int)
    h, w = patch.shape
    for i in range(h):
        for j in range(w - 1):  # Right neighbor
            ref = patch[i, j]
            neigh = patch[i, j + 1]
            co_matrix[ref, neigh] += 1
    return co_matrix

co_matrix = compute_cooccurrence_matrix(normalized_patch)

# --- 1. Show original image with patch box ---
fig, ax = plt.subplots()
ax.imshow(image_np)
rect = patches.Rectangle((col, row), patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title("Original Image with 10x10 Patch Highlighted")
plt.axis('off')
plt.savefig("original_with_patch.png")
plt.show()

# --- 2. Show 10x10 patch (Y channel) ---
plt.imshow(patch_y, cmap='gray')
plt.title("10x10 Patch (Y channel grayscale)")
plt.colorbar()
plt.axis('off')
plt.savefig("patch_y_channel.png")
plt.show()

# --- 3. Print normalized Y values ---
print("🔢 Normalized Y Channel Patch (values from 1 to 10):")
print(normalized_patch)

# --- 4. Show normalized patch with pixel values labeled ---
fig, ax = plt.subplots()
im = ax.imshow(normalized_patch, cmap='viridis')

# Loop over data dimensions and create text annotations.
for i in range(patch_size):
    for j in range(patch_size):
        ax.text(j, i, str(normalized_patch[i, j]),
                ha="center", va="center", color="white", fontsize=8)

ax.set_title("Normalized 10x10 Patch with Pixel Values")
plt.colorbar(im, ax=ax)
plt.axis('off')
plt.savefig("normalized_patch_labeled.png")
plt.show()

# --- 5. Show co-occurrence matrix heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(co_matrix, annot=True, fmt='d', cmap='magma', cbar_kws={'label': 'Count'})
plt.title("Co-Occurrence Matrix\n(Y channel, Horizontal 0° Offset, 1 Pixel)", fontsize=14)
plt.xlabel("Neighbor pixel value", fontsize=12)
plt.ylabel("Reference pixel value", fontsize=12)
plt.xticks(np.arange(11) + 0.5, labels=np.arange(11), rotation=0)
plt.yticks(np.arange(11) + 0.5, labels=np.arange(11), rotation=0)
plt.tight_layout()
plt.savefig("cooccurrence_heatmap.png")
plt.show()
