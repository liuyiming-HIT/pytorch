import torch
import torch.nn as nn

x = torch.tensor([[0.], [1.], [2.], [3.]])
y = torch.tensor([[1.], [3.], [5.], [7.]])

# 使用相同随机种子，获得相同的初始参数
torch.manual_seed(0)
model_adam = nn.Linear(1, 1)

torch.manual_seed(0)
model_sgd = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer_adam = torch.optim.Adam(
    model_adam.parameters(), lr=0.01
)
optimizer_sgd = torch.optim.SGD(
    model_sgd.parameters(), lr=0.05
)

for epoch in range(300):
    # Adam 更新自己的模型
    pred_adam = model_adam(x)
    loss_adam = criterion(pred_adam, y)

    optimizer_adam.zero_grad()
    loss_adam.backward()
    optimizer_adam.step()

    # SGD 更新另一个模型
    pred_sgd = model_sgd(x)
    loss_sgd = criterion(pred_sgd, y)

    optimizer_sgd.zero_grad()
    loss_sgd.backward()
    optimizer_sgd.step()

    if (epoch + 1) % 100 == 0:
        print(
            epoch + 1,
            "Adam:", loss_adam.item(),
            "SGD:", loss_sgd.item()
        )

# 训练结束后，重新计算最终损失
final_adam = criterion(model_adam(x), y)
final_sgd = criterion(model_sgd(x), y)

print("Adam final loss:", final_adam.item())
print("SGD final loss:", final_sgd.item())

print("Adam parameters:",
      model_adam.weight.item(),
      model_adam.bias.item())

print("SGD parameters:",
      model_sgd.weight.item(),
      model_sgd.bias.item())
