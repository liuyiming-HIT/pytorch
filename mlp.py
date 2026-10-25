import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. 生成训练数据
x = torch.linspace(-3, 3, 120).unsqueeze(1)
y = 0.5 * x**2 - x + 1

# 2. 创建 MLP
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 3. 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)

# 4. 训练
for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(epoch + 1, loss.item())

# 5. 测试
x_test = torch.tensor([[-2.], [0.], [2.]])

pred_test = model(x_test)
target_test = 0.5 * x_test**2 - x_test + 1

print("预测值：")
print(pred_test.detach())
print(model[0].weight)
print(model[2].weight)

print("真实值：")
print(target_test)
