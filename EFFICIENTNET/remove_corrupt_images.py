import os
import shutil
from PIL import Image

root = "."  # current folder
corrupt_dir = "./corrupted_files"
os.makedirs(corrupt_dir, exist_ok=True)

for folder, _, files in os.walk(root):
    for file in files:
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"Moving corrupted file: {path}")
            shutil.move(path, corrupt_dir)
