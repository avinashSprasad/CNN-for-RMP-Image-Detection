#import statements
import os
import gc # clear up memory on the gpu
import torch
import wandb
import torch.nn as nn
import torch.optim as optim #optimize waits while training
from PIL import UnidentifiedImageError # used for SafeImagefolder
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms # transforms image

TRAIN_MODEL = True  # ← Set to True if you want to train the model
MODEL_PATH = "resnet_model.pth"#location of stored model

transform = transforms.Compose([ #just creates a list of all the tranformations needed
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # changes image from png to pytorch tensor (usable data for neural network (rgb -> 0-1))
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])# Normalizes each RGB channel to match ImageNet stats for stable and consistent training (subtract mean divide by standard deviation)

])

from torchvision.datasets import ImageFolder
# inheritance |
# Recursively loading the next image on error avoids preprocessing the entire dataset and prevents crashes from corrupted files.
# It also maintains consistent batch sizes and ensures smooth training, resulting in faster overall workflow.

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except UnidentifiedImageError:
            print(f"⚠️ Skipping unreadable image: {self.imgs[index][0]}")
            return self.__getitem__((index + 1) % len(self.imgs))

#just creates a wandb run, sets the name, puts in deepfake detection folder, and has the entity, then sets the architecture
if TRAIN_MODEL:
    wandb.init(
        project="deepfake-detection",
        name="resnet-baseline",
        entity="avisprasad2009-ucsb",
        config={
            "epochs": 7,
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
# only needed, because when not training the orignal is in wandb, but now you can't call it so we create htis
config = wandb.config if TRAIN_MODEL else {
    "batch_size": 128,
    "learning_rate": 0.001
}

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

data_dir = "/home/avinash/dataDetection/genimage/stableDiffusion/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/" # defines where images are formed
full_dataset = SafeImageFolder(root=data_dir, transform=transform) # loads all of them throught the transformes and safe image folders so we can use it
#splits dataset up - labels are names of folders (ai,nature)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size]) # splits the dataset - redudant, can use either one

train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True) # loads the data into batches, after each epoch it shuffles so there is randomization and no pretrained assumptions
val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
class_names = full_dataset.classes # extracts the class labels (ai, natue)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # Parameters:
        # in_channels: number of input channels to the block (e.g. 64, 128) — passed from _make_layer in CustomResNet
        # out_channels: number of output channels — passed from _make_layer to define feature depth
        # stride: controls downsampling — usually 1, but 2 when reducing spatial size (e.g., from 56x56 → 28x28)
        # downsample: a 1x1 conv block used when input and output dimensions don't match — passed from _make_layer

        # This block allows building a flexible residual unit that can preserve or downsample input resolution.
        # Downsampling ensures that input and output shapes are compatible for the skip connection.

        # First convolutional layer: 3x3 kernel, optional downsampling via stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # Padding=1 keeps spatial size constant when stride=1. Stride=2 reduces it.
            # bias=False because BatchNorm follows the convolution and absorbs the bias term.
            nn.BatchNorm2d(out_channels),
            # BatchNorm stabilizes gradients and allows for faster convergence.
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
    """
    The last two parts of the ResidualBlock are the skip connection and the final activation. 
    If the input and output shapes don’t match (for example, if we changed the image size or number of filters),
    we use a special layer called 'downsample' to resize the original input so it can be added correctly to the output.
    Then we add the input (identity) to the output of the convolutions — this is called a skip connection.
    It helps the model learn better and not "forget" important features from earlier layers.
    Finally, we use ReLU (a simple activation function) to make sure the output stays non-linear,
    which helps the model learn more complex patterns in the image.
    This whole process is key to why ResNet works well, because it allows deep models to train without getting stuck.
    """


class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2): # blocks is using residual block class, layers is amount of times to run residual block in each layer, and num class means tow classifications (real fake)
        super(CustomResNet, self).__init__() # calls contrctor of parent class
        self.in_channels = 64#how many inputs (different patterns like edges or rgb)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # Input has 3 RGB channels; conv layer applies 64 filters with a 7x7 kernel, stride 2 for downsampling, and padding 3 to preserve spatial size; bias is disabled because BatchNorm2d(64) normalizes the feature maps to stabilize training; ReLU adds non-linearity; MaxPool2d with 3x3 kernel, stride 2, padding 1 further downsamples by taking max values to reduce noise and improve efficiency.
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # runs the block, number of output channels, layres is the number of blocks in the list of layres, downsampling is happenign
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        # This makes each feature map into just one number by averaging it.
        # It’s good because no matter what size the input is, it makes the data smaller and easier to work with.

        self.fc = nn.Linear(512, num_classes)          
        # This takes those numbers and turns them into scores for each class.
        # It’s important because this is how the model decides what the picture shows.


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
        '''
        Builds a group (layer) of residual blocks for the ResNet model.

        If the stride is not 1 or the number of input channels is different from 
        the output channels, it creates a downsample layer to adjust the input 
        so the skip connection can be added properly.

        The first block in the group uses this downsample (if needed) and may 
        also reduce the spatial size (downsample).

        The rest of the blocks keep the same number of channels and spatial size.

        All blocks are combined into a sequential container to run them in order.
        '''

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #just inputs each image and runs it throught each layer till the end
        out = self.avg_pool(out)#This step squishes each feature map down to just one number by averaging all the values.
        out = torch.flatten(out, 1)#This turns all those numbers into a long list (a flat vector) for each image. |flatten(..., 1) turns it into [batch_size, channels * height * width]
        return self.fc(out)#This last part takes the list of numbers and turns it into scores for each class (like "AI" or "Nature").

model = CustomResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)  # Builds the ResNet model and sends it to GPU or CPU so it can run faster.
criterion = nn.CrossEntropyLoss()  # Sets the loss function for classification; helps the model learn to predict the right class.
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])  # Uses the Adam optimizer to update model weights during training.

if TRAIN_MODEL:
    for epoch in range(config["epochs"]):
        model.train()  # Puts the model in training mode (enables dropout, batch norm updates, etc.).
        running_loss = 0.0  
        correct = 0  
        total = 0  

        for step, (images, labels) in enumerate(train_loader):  # Loops through the training dataset in batches.
            images, labels = images.to(device), labels.to(device)  # Moves the data to the device (GPU or CPU).
            optimizer.zero_grad()  # Clears old gradients so they don’t build up.
            outputs = model(images)  # Runs the model to get predictions for the current batch.
            loss = criterion(outputs, labels)  # Calculates how far the predictions are from the true labels.
            loss.backward()  # Computes the gradients (how to change the weights to improve).
            optimizer.step()  # Updates the model’s weights based on the gradients.

            _, predicted = torch.max(outputs, 1)  # Takes the highest score as a prediction
            correct += (predicted == labels).sum().item()  # Counts correct predictions in this batch; .sum() totals True values, .item() converts it from tensor to Python number.
    total += labels.size(0)  # Adds the number of samples in the current batch to the total seen so far; labels.size(0) gives batch size.
    step_acc = correct / total 
    wandb.log({  # Logs metrics for the current training step to Weights & Biases for live tracking and graphs.
        "train_step_loss": loss.item(),  # Converts the batch loss from a tensor to a float for logging.
        "train_step_accuracy": step_acc,  # Logs current step accuracy to track how well the model is learning.
        "epoch": epoch + 1  # Records which epoch this step belongs to (adding 1 for human-friendly count).
    })

    running_loss += loss.item() * images.size(0)  # Accumulates total loss by multiplying loss per sample by batch size.

    epoch_loss = running_loss / len(train_loader.dataset)  #average loss for this epoch
    epoch_acc = correct / total  # Calculates full training accuracy after all batches in the epoch.

    model.eval()  
    val_loss = 0.0  
    val_correct = 0  
    val_total = 0  
    with torch.no_grad():  # Turns off gradient tracking to save memory and speed up validation.
        for images, labels in val_loader:  # Loops over batches in the validation dataset.
            images, labels = images.to(device), labels.to(device)  # Moves validation data to the correct device (GPU or CPU).
            outputs = model(images)  # Gets model predictions for this batch.
            loss = criterion(outputs, labels)  # Calculates the loss for this batch of validation data.

            val_loss += loss.item() * images.size(0)  # Accumulates total validation loss.
            _, predicted = torch.max(outputs, 1)  # Selects class with highest score for each prediction.
            val_correct += (predicted == labels).sum().item()  # Adds correct predictions for this batch to the total.
            val_total += labels.size(0)  # Adds number of samples in the batch to the total.

            del images, labels, outputs  # Frees up memory from variables no longer needed.
            torch.cuda.empty_cache()  # Clears unused memory on GPU (safe housekeeping).
            gc.collect()  # Forces garbage collection to free up Python memory.

    val_loss /= len(val_loader.dataset)  # Averages total validation loss over all validation samples.
    val_acc = val_correct / val_total  # Calculates overall validation accuracy.

    wandb.log({  # Logs final metrics for the epoch to Weights & Biases.
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")  # Prints summary of epoch performance to terminal.

    torch.save(model.state_dict(), f"resnet_epoch_{epoch+1}.pth")  # Saves the model’s weights for this epoch so training can be resumed or used later.

# when trained on 6 epochs :  TEST SET — Loss: 0.0891, Accuracy: 0.9769