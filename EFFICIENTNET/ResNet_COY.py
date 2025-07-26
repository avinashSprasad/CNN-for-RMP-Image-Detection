import os
import gc
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError
from Co_Occurrence import CoOccurenceProcessor
import torch.nn.functional as F

# -------------------- WANDB SETUP --------------------
wandb.init(
    project="ResNET_CORGB_BIGGAN",              # ✅ New wandb project name
    name="ResNET_COY",                   # ✅ Run name
    mode="online",
    settings=wandb.Settings(console="off")
)

# -------------------- CONFIG --------------------
config = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
    "image_size": 224,
    "save_dir": "/home/avinash/detectionCode/RESNET"
}
wandb.config.update(config)

# -------------------- DEVICE --------------------
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

# -------------------- SAFE LOADER --------------------
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

        if self.transform is not None:
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

# Calculate split sizes
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Perform split
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

print("✅ Data loaded safely")

# -------------------- CUSTOM RESNET --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.in_channels = 64

        self.co_occurrence = CoOccurenceProcessor(num_levels=256)

        # 🔸 Modified: Input channels changed from 3 → 6 because co-occurrence returns 6 channels
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.co_occurrence(x)                    # Shape: [B, 6, H, W]

        # 🔸 Optional: Downsample spatial resolution to reduce memory (can be removed)
        x = F.adaptive_avg_pool2d(x, (32, 32))       # Helps with CUDA timeout if needed

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CustomResNet(ResidualBlock, [2, 2, 2, 2], num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

from tqdm import tqdm

 # 🔴 Limit batches per epoch
batch_limit =400
# 🔵 Outer tqdm for epochs
for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress", position=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 🟢 TRAINING LOOP WITH TQDM
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
        _, preds = torch.max(outputs, 1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / batch_limit
    train_acc = 100. * correct / total

    # -------------------- VALIDATION --------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", position=2, leave=False)):
            if i >= batch_limit:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
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

    print(f"[Epoch {epoch + 1}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # -------------------- SAVE MODEL --------------------
    os.makedirs(config["save_dir"], exist_ok=True)

    # Save model for this epoch
    epoch_save_name = f"ResNET_epoch_BigGAN_{epoch + 1}.pth"
    epoch_save_path = os.path.join(config["save_dir"], epoch_save_name)
    torch.save(model.state_dict(), epoch_save_path)
    print(f"💾 Epoch {epoch+1} model saved to {epoch_save_path}")

    # Save final model on last epoch
    if epoch == config["epochs"] - 1:
        final_save_name = "ResNET_GAN_CORGB.pth"
        final_save_path = os.path.join(config["save_dir"], final_save_name)
        torch.save(model.state_dict(), final_save_path)
        print(f"🏁 Final model saved to {final_save_path}")

    # Optional: free memory
    gc.collect()

# -------------------- TESTING (after all epochs) --------------------
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
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
