
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
from Co_OccurrenceCopy import CoOccurenceProcessor


# -------------------- WANDB SETUP --------------------
wandb.init(
    project="EfficientNetWithCbCr",              # ✅ New wandb project name
    name="EfficientNetWithCBCR",                   # ✅ Run name
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

# -------------------- MODEL --------------------
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Depthwise convolution
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Projection phase
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        else:
            return out

import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    def __init__(self, co_occurrence, num_classes=2):
        super(CustomEfficientNet, self).__init__()
        self.co_occurrence = co_occurrence

        self.stem = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = MBConvBlock(32, 64, expansion_factor=6, stride=2)
        self.block2 = MBConvBlock(64, 128, expansion_factor=6, stride=2)
        self.block3 = MBConvBlock(128, 128, expansion_factor=6, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # We'll set this later after we know the flattened feature size
        self._fc_in_features = None
        self.fc = None

    def _initialize_fc(self, x):
        """Call this once to initialize the fully connected layer with correct shape."""
        with torch.no_grad():
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x)
            x = self.flatten(x)
            self._fc_in_features = x.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(self._fc_in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )

    def forward(self, x):
        x = self.co_occurrence(x)

        # Initialize FC layer once, on the correct device
        if self.fc is None:
            self._initialize_fc(x.to(next(self.parameters()).device))

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)








print("Starting loading model.")

co_occurrence = CoOccurenceProcessor()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = CustomEfficientNet(co_occurrence, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("Finished loading model")




# ✅ Create save directory once, before training loop
os.makedirs(config["save_dir"], exist_ok=True)

best_train_acc = 0.0
best_val_acc = 0.0
batch_limit = 1000

# 🔵 Outer tqdm for epochs
for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress", position=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 🟢 TRAINING LOOP
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

    # -------------------- CONDITIONAL SAVE --------------------
    if train_acc > best_train_acc and val_acc > best_val_acc:
        best_train_acc = train_acc
        best_val_acc = val_acc

        best_save_name = "efficientNet_COYCBCR_BIGGAN_Validation_improvement_best.pth"
        best_save_path = os.path.join(config["save_dir"], best_save_name)
        torch.save(model.state_dict(), best_save_path)
        print(f"💾 Best model saved (Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%) → {best_save_path}")

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
'''
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
from Co_OccurrenceCopy import CoOccurenceProcessor


# -------------------- WANDB SETUP --------------------
wandb.init(
    project="EfficientNetWithCbCr",              # ✅ New wandb project name
    name="EfficientNetPooling8",                   # ✅ Run name
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

# -------------------- MODEL --------------------
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Depthwise convolution
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Projection phase
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        else:
            return out

import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    def __init__(self, co_occurrence, num_classes=1):
        super(CustomEfficientNet, self).__init__()
        self.co_occurrence = co_occurrence

        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = MBConvBlock(32, 64, expansion_factor=6, stride=2)
        self.block2 = MBConvBlock(64, 128, expansion_factor=6, stride=2)
        self.block3 = MBConvBlock(128, 128, expansion_factor=6, stride=1)

        # Combined pooling layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.co_occurrence(x)    # [B, 4, H, W]
        x = self.stem(x)             # [B, 32, H/2, W/2]
        x = self.block1(x)           # [B, 64, H/4, W/4]
        x = self.block2(x)           # [B, 128, H/8, W/8]
        x = self.block3(x)           # [B, 128, H/8, W/8]

        # Combined global pooling
        x_avg = self.avgpool(x)      # [B, 128, 1, 1]
        #x_max = self.maxpool(x)      # [B, 128, 1, 1]
        x = (x_avg) / 2      # [B, 128, 1, 1]

        x = self.flatten(x)          # [B, 128]
        return self.fc(x)            # [B, num_classes]





print("Starting loading model.")

co_occurrence = CoOccurenceProcessor()
model = CustomEfficientNet(co_occurrence=co_occurrence, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("Finished loading model")




# ✅ Create save directory once, before training loop
os.makedirs(config["save_dir"], exist_ok=True)

best_train_acc = 0.0
best_val_acc = 0.0
batch_limit = 1000

# 🔵 Outer tqdm for epochs
for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress", position=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 🟢 TRAINING LOOP
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

    # -------------------- CONDITIONAL SAVE --------------------
    if train_acc > best_train_acc and val_acc > best_val_acc:
        best_train_acc = train_acc
        best_val_acc = val_acc

        best_save_name = "efficientNet_CORGB_BIGGAN_best.pth"
        best_save_path = os.path.join(config["save_dir"], best_save_name)
        torch.save(model.state_dict(), best_save_path)
        print(f"💾 Best model saved (Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%) → {best_save_path}")

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
'''

'''
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
    project="EfficientNetPooling8",              # ✅ New wandb project name
    name="EfficientNetImprovingAccuracy",                   # ✅ Run name
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
    transforms.Resize((config["image_size"], config["image_size"])),           # Uniform input size
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),       # Mirror textures
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),           # Orientation variation
    transforms.RandomApply([                                                   
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
    ], p=0.4),                                                                 # Illumination variation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),                 # Texture smoothing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                           # Standard ImageNet normalization
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

# -------------------- MODEL --------------------
import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Depthwise convolution
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Projection phase
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        else:
            return out

import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    def __init__(self, co_occurrence, num_classes=1):
        super(CustomEfficientNet, self).__init__()
        self.co_occurrence = co_occurrence

        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = MBConvBlock(32, 64, expansion_factor=6, stride=2)
        self.block2 = MBConvBlock(64, 128, expansion_factor=6, stride=2)
        self.block3 = MBConvBlock(128, 128, expansion_factor=6, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.co_occurrence(x)    # [B, 4, H, W]
        x = self.stem(x)             # [B, 32, H/2, W/2]
        x = self.block1(x)           # [B, 64, H/4, W/4]
        x = self.block2(x)           # [B, 128, H/8, W/8]
        x = self.block3(x)           # [B, 128, H/8, W/8]

        x = self.avgpool(x)          # [B, 128, 1, 1]
        x = self.flatten(x)          # [B, 128]
        return self.fc(x)            # [B, num_classes]





print("Starting loading model.")

co_occurrence = CoOccurenceProcessor()
model = CustomEfficientNet(co_occurrence=co_occurrence, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("Finished loading model")




# ✅ Create save directory once, before training loop
os.makedirs(config["save_dir"], exist_ok=True)

best_train_acc = 0.0
best_val_acc = 0.0
batch_limit = 1000

# 🔵 Outer tqdm for epochs
for epoch in tqdm(range(config["epochs"]), desc="Epoch Progress", position=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 🟢 TRAINING LOOP
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

    # -------------------- CONDITIONAL SAVE --------------------
    if train_acc > best_train_acc and val_acc > best_val_acc:
        best_train_acc = train_acc
        best_val_acc = val_acc

        best_save_name = "efficientNet_COY_BIGGAN_best.pth"
        best_save_path = os.path.join(config["save_dir"], best_save_name)
        torch.save(model.state_dict(), best_save_path)
        print(f"💾 Best model saved (Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%) → {best_save_path}")

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


'''