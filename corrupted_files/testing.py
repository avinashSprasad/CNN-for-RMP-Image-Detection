import os
import torch
import wandb
import PIL
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError

print("starting testing")

# Initialize wandb
wandb.init(
    project="deepfake-detection",
    name="resnet-eval",
    entity="avisprasad2009-ucsb"
)

# Custom SafeImageFolder to skip unreadable/corrupt images
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"⚠️ Skipping unreadable image: {self.imgs[index][0]}")
            return self.__getitem__((index + 1) % len(self.imgs))

# Configuration
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 6
}

# Transforms (resize, tensor conversion, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Dataset path
data_dir = "/home/avinash/dataDetection/GenImage/train"

# Load dataset with safe loader and apply transforms
full_dataset = SafeImageFolder(root=data_dir, transform=transform)

# Split dataset into train, val, test (only test used here)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

# DataLoader for test set
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

# Residual Block for ResNet
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

# Custom ResNet Model
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

# Load model and weights
MODEL_PATH = "resnet_model.pth"
model = CustomResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# Index-to-class mapping
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

# Test evaluation loop
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Calculate original dataset indices for this batch
        batch_start = batch_idx * config["batch_size"]
        batch_end = batch_start + images.size(0)
        batch_indices = test_ds.indices[batch_start:batch_end]

        for i, idx in enumerate(batch_indices):
            img_path, _ = full_dataset.samples[idx]
            true_label = idx_to_class[labels[i].item()]
            pred_label = idx_to_class[predicted[i].item()]
            result = "✅ CORRECT" if true_label == pred_label else "❌ WRONG"
            print(f"{img_path} — Predicted: {pred_label} | Actual: {true_label} → {result}")

            # Show image
            img = images[i].cpu().clone()
            img = img.permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = img.clamp(0, 1)
            plt.imshow(img)
            plt.title(f"{result}\nPredicted: {pred_label} | Actual: {true_label}")
            plt.axis("off")
            plt.show()

        del images, labels, outputs
        torch.cuda.empty_cache()

test_loss /= len(test_loader.dataset)
test_acc = correct / total

# Log to wandb
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_acc
})

print(f"TEST SET — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")


#TEST SET — Loss: 0.0212, Accuracy: 0.9934| TEST SET — Loss: 0.0189, Accuracy: 0.9939
#cdTEST SET — Loss: 0.0195, Accuracy: 0.9937