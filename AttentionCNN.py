import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# 1️⃣ Squeeze-and-Excitation Block
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ----------------------------
# 2️⃣ Bottleneck Residual Block with SE
# ----------------------------
class BottleneckSEBlock(nn.Module):
    """
    Bottleneck Residual Block with SE attention
    Structure: 1x1 reduce -> 3x3 conv -> 1x1 expand
    """
    expansion = 4  # final channels = out_channels * expansion
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        mid_channels = out_channels
        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 conv
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 expand
        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        # SE block
        self.se = SEBlock(out_channels * self.expansion, reduction)

        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

# ----------------------------
# 3️⃣ Advanced CNN with BottleneckSE
# ----------------------------
class ResSE_Bottleneck_CNN(nn.Module):
    """
    Advanced CNN for CIFAR-10 with bottleneck residual blocks + SE
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Layer config: (blocks, out_channels)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64 * BottleneckSEBlock.expansion, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128 * BottleneckSEBlock.expansion, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256 * BottleneckSEBlock.expansion, 512, blocks=3, stride=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512 * BottleneckSEBlock.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BottleneckSEBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BottleneckSEBlock(out_channels * BottleneckSEBlock.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ----------------------------
# 1️⃣ Data preparation for CIFAR-10
# ----------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # random crop + padding
    transforms.RandomHorizontalFlip(),      # random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------
# 2️⃣ Define the model
# ----------------------------
# (Using the ResSE_Bottleneck_CNN from previous step)
# Make sure this class is defined above in the same file or imported

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResSE_Bottleneck_CNN().to(device)

# ----------------------------
# 3️⃣ Loss function and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# Cosine annealing LR scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ----------------------------
# 4️⃣ Training loop
# ----------------------------
num_epochs = 50  # can increase for 99%+ accuracy
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 49:  # print every 50 batches
            print(f"[Epoch {epoch}, Batch {batch_idx+1}] Loss: {running_loss / 50:.4f}")
            running_loss = 0.0

    # Step LR scheduler after each epoch
    scheduler.step()

    # ----------------------------
    # 5️⃣ Evaluate on test set
    # ----------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch} Test Accuracy: {100*correct/total:.2f}%\n")

# ----------------------------
# 6️⃣ Save the trained model
# ----------------------------
torch.save(model.state_dict(), "resse_bottleneck_cifar10.pth")
print("Model saved as resse_bottleneck_cifar10.pth")
