import os
import gc
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError, Image
from tqdm import tqdm  # 🔴 ADDED
# -------------------- WANDB SETUP --------------------
wandb.init(project="efficientnet-deepfake", name="custom-efficientnet")

# ... your imports and other code ...

# -------------------- CONFIG --------------------
config = {
    "epochs": 5,
    "batch_size": 32,  # Reduced from 32 to 8
    "lr": 1e-4,
    "image_size": 224,
    "save_dir": "/home/avinash/detectionCOde/EFFICIENTNET"
}
wandb.config.update(config)

# -------------------- DEVICE --------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Clear CUDA cache before training
if device.type == 'cuda':
    torch.cuda.empty_cache()

# ... rest of your code unchanged ...

# -------------------- DEVICE --------------------


# -------------------- SAFE IMAGEFOLDER WITH PRE-FILTER --------------------
'''
class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        good_imgs = []
        print("🧹 Filtering and deleting unreadable images...")

        # ✅ tqdm shows progress over image paths
        for path, label in tqdm(self.imgs, desc="Checking images"):
            try:
                with Image.open(path) as img:
                    img.verify()  # ✅ Check image file integrity
                good_imgs.append((path, label))
            except Exception as e:
                print(f"❌ Corrupted image detected: {path} ({e})")
                try:
                    os.remove(path)  # ✅ Permanently delete the corrupted image
                    print(f"🗑️ Deleted: {path}")
                except Exception as del_err:
                    print(f"⚠️ Failed to delete {path}: {del_err}")

        # ✅ Keep only valid images
        self.imgs = good_imgs
        self.samples = good_imgs  # ImageFolder uses .samples internally

    def __getitem__(self, index):
        # ✅ No try-except needed — only good files remain
        return super().__getitem__(index)
'''
# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- DATA --------------------
data_dir = "/home/avinash/dataDetection/genimage/stableDiffusion/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/"
dataset = ImageFolder(data_dir, transform=transform)

print("Found classes:", dataset.classes)
print(f"Number of images: {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

# -------------------- MODEL --------------------
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_residual = in_channels == out_channels

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out += x
        return out

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = MBConvBlock(32, 32)
        self.block2 = MBConvBlock(32, 64)
        self.block3 = MBConvBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = CustomEfficientNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])



batch_limit = 20  # 🔴 ADDED: Limit batches per epoch for shorter training

for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 🟢 CHANGED: tqdm and batch limit
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
        if i >= batch_limit:
            break  # 🔴 ADDED: shorten the epoch by cutting off batch count

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / batch_limit  # 🟢 CHANGED denominator
    train_acc = 100. * correct / total

    # -------------------- VALIDATION --------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # 🟢 CHANGED: tqdm and batch limit
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)):
            if i >= batch_limit:
                break  # 🔴 ADDED: shorten the epoch for validation too

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= batch_limit  # 🟢 CHANGED denominator
    val_acc = 100. * val_correct / val_total

    # -------------------- LOGGING --------------------
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"[Epoch {epoch + 1}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # -------------------- SAVE MODEL --------------------
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(config["save_dir"], f"efficientnet_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), save_path)
    gc.collect()

# -------------------- FINISH --------------------
wandb.finish()
