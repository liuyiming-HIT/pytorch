import torch
import torch.nn as nn
x = torch.randn(4, 3, 64, 64)
conv=nn.Conv2d(3,8,3,1,1)
y=conv(x)
print(y.shape)
relu=nn.ReLU()
z=relu(y)
print(z.shape)
mp=nn.MaxPool2d(2,2)
d=mp(z)
print(d.shape)



