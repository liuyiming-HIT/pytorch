import torch
import torch.nn as nn

x = torch.tensor([[1.], [2.], [3.], [4.]])
y = torch.tensor([[3.], [5.], [7.], [9.]])

model=nn.Linear(1,1)
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.5)

for epoch in range(300):
    pred=model(x)
    loss=criterion(pred,y)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if (epoch+1) %5 == 0:
        print(epoch+1,loss.item())
print(model.weight.item(),model.bias.item())

