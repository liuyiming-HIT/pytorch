import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(0)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 1. 加载 MNIST
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_set, val_set = random_split(
    dataset, [54000, 6000],
    generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(
    train_set, batch_size=64, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=64, shuffle=False
)

# 2. CNN 模型
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 3. 训练与验证
for epoch in range(3):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predicted = logits.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {train_loss/len(train_set):.4f}, "
        f"Val Accuracy: {correct/total:.4f}"
    )

# 4. 保存权重
torch.save(model.state_dict(), "mnist_cnn.pth")
