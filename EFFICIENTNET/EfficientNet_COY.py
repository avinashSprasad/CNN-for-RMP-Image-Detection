import sys
import os
import gc
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import wandb

# Make sure Co_Occurrence module is available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Co_Occurrence import CoOccurenceProcessor

# -------------------- WANDB SETUP --------------------
wandb.init(
    project="efficientNet_CORGB",              # ✅ New wandb project name
    name="efficientNet_COY",                   # ✅ Run name
    mode="online",
    settings=wandb.Settings(console="off")
)

# -------------------- CONFIG --------------------
config = {
    
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-4,
    "image_size": 224,
    "save_dir": "/home/avinash/detectionCOde/EFFICIENTNET"
}
wandb.config.update(config)

# -------------------- DEVICE --------------------
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- DATA --------------------
#data_dir = "/home/avinash/dataDetection/genimage/stableDiffusion/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/"
data_dir = "/home/avinash/dataDetection/GenImage/imagenet_ai_0419_biggan/train/"
dataset = ImageFolder(data_dir, transform=transform)

print("Found classes:", dataset.classes)
print(f"Number of images: {len(dataset)}")

dataset_size = len(dataset)

# Calculate split sizes
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size  # remainder

# Perform split
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=2, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

        # Apply co-occurrence on raw image: [B, 3, H, W]
        self.co_occurrence = CoOccurenceProcessor(num_levels=num_levels)

        # Co-occurrence output shape: [B, 2*3, 256, 256]
        cooc_channels = 2 * 3  # 2 directions, 3 channels (Y or RGB)

        self.stem = nn.Sequential(
            nn.Conv2d(cooc_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = MBConvBlock(32, 32)
        self.block2 = MBConvBlock(32, 64)
        self.block3 = MBConvBlock(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1))  # Reduce for final FC

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        co_feat = self.co_occurrence(x)  # [B, 6, 256, 256]

        # Downsample co-occurrence to reduce huge spatial size (optional but recommended)
        co_feat = F.adaptive_avg_pool2d(co_feat, (32, 32))  # for example

        x = self.stem(co_feat)           # e.g. [B, 32, 16, 16]
        x = self.block1(x)

        x = self.pool(x)                 # [B, 32, 1, 1]

        x = self.block2(x)              # Will keep spatial 1x1 because block conv stride=1 and padding=1 might not preserve shape? Check below.
        x = self.pool(x)                 # Probably redundant pooling here if spatial is already 1x1, but safe.

        x = self.block3(x)
        x = self.pool(x)

        x = self.flatten(x)              # [B, 128]

        return self.fc(x)


print("Starting loading model.")

model = CustomEfficientNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("Finished loading model")

# -------------------- TRAINING --------------------
batch_limit = 63  # Optional: speed up debugging

# ✅ Create save directory once, before training loop
os.makedirs(config["save_dir"], exist_ok=True)

# ✅ Training loop
for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
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

    # -------------------- VALIDATION --------------------
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

    # -------------------- LOGGING --------------------
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"[Epoch {epoch+1}] ✅ Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # -------------------- SAVE MODEL --------------------
    # 💾 Save model for this epoch
    epoch_save_name = f"efficientNet_CORGB_epoch_{epoch+1}.pth"
    epoch_save_path = os.path.join(config["save_dir"], epoch_save_name)
    torch.save(model.state_dict(), epoch_save_path)
    print(f"💾 Epoch {epoch+1} model saved to {epoch_save_path}")

    # 💾 Save final model with simplified name on last epoch
    if epoch == config["epochs"] - 1:
        final_save_name = "efficientNet_CORGB.pth"
        final_save_path = os.path.join(config["save_dir"], final_save_name)
        torch.save(model.state_dict(), final_save_path)
        print(f"🏁 Final model saved to {final_save_path}")

    # Optional: free memory
    gc.collect()

# -------------------- TESTING --------------------
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
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

# -------------------- FINISH --------------------
wandb.finish()
