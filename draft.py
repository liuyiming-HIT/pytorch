import torch 
import torch.nn as nn
model=nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),#(8,8,80,120)
    nn.ReLU(),
    nn.MaxPool2d(2),#(8,8,40,60)

    nn.Conv2d(8, 16, kernel_size=3, padding=1),#(8,16,40,60)
    nn.ReLU(),
    nn.MaxPool2d(2),#(8,16,20,30)

    nn.AdaptiveAvgPool2d((1, 1)),#(8,16,1,1)
    nn.Flatten()#(8,16)
)
depth=torch.randn(8,1,80,120)
pred=model(depth)

import torch 
import torch.nn as nn
model=nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),#(8,8,80,120)
    nn.ReLU(),
    nn.MaxPool2d(2),#(8,8,40,60)

    nn.Conv2d(8, 16, kernel_size=3, padding=1),#(8,16,40,60)
    nn.ReLU(),
    nn.MaxPool2d(2),#(8,16,20,30)

    nn.AdaptiveAvgPool2d((1, 1)),#(8,16,1,1)
    nn.Flatten()#(8,16)
)
depth=torch.randn(2,1,80,120)
pred=model(depth)
print(pred.shape,depth.shape)
