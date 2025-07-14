'''
import os
import gc
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

=
TRAIN_MODEL = False  # ← Set to True if you want to train the model
MODEL_PATH = "resnet_model.pth"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"⚠️ Skipping unreadable image: {self.imgs[index][0]}")
            return self.__getitem__((index + 1) % len(self.imgs))


if TRAIN_MODEL:
    wandb.init(
        project="deepfake-detection",
        name="resnet-baseline",
        entity="avisprasad2009-ucsb",
        config={
            "epochs": 5,
            "learning_rate": 0.001,
            "architecture": "CustomResNet",
            "optimizer": "Adam",
            "batch_size": 128
        }
    )
else:
    wandb.init(
        project="deepfake-detection",
        name="resnet-eval",
        entity="avisprasad2009-ucsb"
    )

config = wandb.config if TRAIN_MODEL else {
    "batch_size": 128,
    "learning_rate": 0.001
}

# ========== DEVICE ==========
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


data_dir = "/home/avinash/dataDetection/GenImage/train"
full_dataset = SafeImageFolder(root=data_dir, transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
class_names = full_dataset.classes

# ========== MODEL ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# ========== INIT MODEL ==========
model = CustomResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# ========== TRAINING ==========
if TRAIN_MODEL:
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            step_acc = correct / total

            wandb.log({
                "train_step_loss": loss.item(),
                "train_step_accuracy": step_acc,
                "epoch": epoch + 1
            })

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        wandb.log({
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save trained model
    torch.save(model.state_dict(), MODEL_PATH)
else:
    # Load pretrained model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

wandb.finish()
'''
import os
import gc
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ========== CONFIG ==========
TRAIN_MODEL = False  # ← Set to True if you want to train the model
MODEL_PATH = "resnet_model.pth"

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== SAFE IMAGEFOLDER ==========
from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"⚠️ Skipping unreadable image: {self.imgs[index][0]}")
            return self.__getitem__((index + 1) % len(self.imgs))

# ========== WANDB INIT ==========
if TRAIN_MODEL:
    wandb.init(
        project="deepfake-detection",
        name="resnet-baseline",
        entity="avisprasad2009-ucsb",
        config={
            "epochs": 5,
            "learning_rate": 0.001,
            "architecture": "CustomResNet",
            "optimizer": "Adam",
            "batch_size": 128
        }
    )
else:
    wandb.init(
        project="deepfake-detection",
        name="resnet-eval",
        entity="avisprasad2009-ucsb"
    )

config = wandb.config if TRAIN_MODEL else {
    "batch_size": 128,
    "learning_rate": 0.001
}

# ========== DEVICE ==========
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ========== DATA ==========
data_dir = "/home/avinash/dataDetection/GenImage/train"
full_dataset = SafeImageFolder(root=data_dir, transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
class_names = full_dataset.classes

# ========== MODEL ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

# ========== INIT MODEL ==========
model = CustomResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# ========== TRAINING ==========
if TRAIN_MODEL:
    # [training loop not shown for brevity]
    torch.save(model.state_dict(), MODEL_PATH)
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

wandb.finish()

# ========== TEST SPLIT EVALUATION ==========
def evaluate_test_split(model):
    print("\n🔍 Evaluating on held-out TEST split from train dataset...")
    correct = 0
    total = 0
    loss_sum = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * images.size(0)

            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = loss_sum / total
    accuracy = correct / total

    print(f"[TEST SPLIT] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# ========== RUN TEST SPLIT EVALUATION ==========
evaluate_test_split(model)
