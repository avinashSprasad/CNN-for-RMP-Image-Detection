import sys
import os
import gc
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure your co_occurrence_rgb.py file is in the same folder or accessible via PYTHONPATH
from co_occurrence_rgb import RGBGLCM

# -------------------- WANDB SETUP --------------------
wandb.init(
    project="mainstream_",          # renamed project
    name="efficientNetCORGB_BIGGAN",
    mode="online",
    settings=wandb.Settings(console="off")
)

# -------------------- CONFIG --------------------
config = {
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-5,
    "image_size": 224,
    "save_dir": "/home/avinash/detectionCOde/EFFICIENTNET"
}
wandb.config.update(config)

# -------------------- DEVICE --------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -------------------- SAFE LOADER --------------------
from PIL import Image, UnidentifiedImageError

def safe_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"⚠️ Skipping unreadable image: {path}")
        return None

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_loader(path)
        while sample is None:
            index = (index + 1) % len(self.samples)
            path, target = self.samples[index]
            sample = safe_loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- DATA --------------------
data_dir = "/home/avinash/dataDetection/GenImage/imagenet_ai_0419_biggan/train/"
dataset = SafeImageFolder(data_dir, transform=transform)

print("Found classes:", dataset.classes)
print(f"Number of images: {len(dataset)}")

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

print("✅ Data loaded safely")

# -------------------- MODEL --------------------
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
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

import torch.nn.functional as F

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2, num_levels=256):
        super().__init__()
        self.num_levels = num_levels
        self.co_occurrence = RGBGLCM(num_levels=num_levels)


        cooc_channels = 2 * 3  # 2 directions * 3 channels (RGB)

        self.stem = nn.Sequential(
            nn.Conv2d(cooc_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = MBConvBlock(32, 32)
        self.block2 = MBConvBlock(32, 64)
        self.block3 = MBConvBlock(64, 128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.co_occurrence(x)  # expect shape [B, 6, 256, 256]
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

print("Starting loading model.")

model = CustomEfficientNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("Finished loading model")

# Create save directory once
os.makedirs(config["save_dir"], exist_ok=True)

batch_limit = 400

for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress", position=0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", position=1, leave=False)):
        if i >= batch_limit:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / batch_limit
    train_acc = 100. * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    print("VALIDATION")

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)):
            if i >= batch_limit:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= batch_limit
    val_acc = 100. * val_correct / val_total

    # Log metrics
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"[Epoch {epoch+1}] ✅ Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save model checkpoint
    epoch_save_name = f"efficientNet_CORGB__BIGGANepoch_{epoch+1}.pth"
    epoch_save_path = os.path.join(config["save_dir"], epoch_save_name)
    torch.save(model.state_dict(), epoch_save_path)
    print(f"💾 Epoch {epoch+1} model saved to {epoch_save_path}")

    if epoch == config["epochs"] - 1:
        final_save_name = "efficientNet_CORGB_BIGGAN.pth"
        final_save_path = os.path.join(config["save_dir"], final_save_name)
        torch.save(model.state_dict(), final_save_path)
        print(f"🏁 Final model saved to {final_save_path}")

    gc.collect()

# Testing
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
print("TESTING")

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        preds = outputs.argmax(dim=1)
        test_correct += preds.eq(labels).sum().item()
        test_total += labels.size(0)

test_loss /= len(test_loader)
test_acc = 100. * test_correct / test_total

print(f"🧪 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_acc
})

wandb.finish()
