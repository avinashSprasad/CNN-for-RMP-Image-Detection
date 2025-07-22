from PIL import Image
import matplotlib.pyplot as plt

def show_images_grid(image_paths):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    for ax, path in zip(axes, image_paths):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(path.split("/")[-1], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# List of 5 image file paths
image_paths = [
    "/home/avinash/dataDetection/GenImage/train/nature/n02497673_5971.JPEG",     # Correct
    "/home/avinash/dataDetection/GenImage/train/nature/n02109047_22700.JPEG",    # Correct
    "/home/avinash/dataDetection/GenImage/train/ai/998_biggan_00102.png",        # Correct
    "/home/avinash/dataDetection/GenImage/train/nature/n01980166_127.JPEG",      # Wrong
    "/home/avinash/dataDetection/GenImage/train/nature/n03729826_13944.JPEG"     # Wrong
]

show_images_grid(image_paths)
