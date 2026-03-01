import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CNN import simplecnn  # 注意：不写 model.CNN

# ===================== #
# 1. 参数设置
# ===================== #
batch_size = 64
num_epochs = 2      # 先来个小的测试一下
learning_rate = 1e-3
num_classes = 10    # ⚠️ CIFAR-10 就是 10 类

# ===================== #
# 2. 数据加载：CIFAR-10
# ===================== #
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(
    root='../data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.CIFAR10(
    root='../data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== #
# 3. 模型、损失函数、优化器
# ===================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = simplecnn(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 小 debug：确认标签范围 & 模型输出
images, labels = next(iter(train_loader))
print("DEBUG -> labels min:", labels.min().item(), "max:", labels.max().item())
with torch.no_grad():
    out = model(images.to(device))
print("DEBUG -> model output shape:", out.shape)

# ===================== #
# 4. 训练循环
# ===================== #
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# ===================== #
# 5. 测试模型
# ===================== #
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# ===================== #
# 6. 保存模型
# ===================== #
torch.save(model.state_dict(), "simplecnn_weights.pth")
print("Model saved as simplecnn_weights.pth ✅")
