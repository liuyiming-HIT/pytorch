class DepthStateFusion(nn.Module):
    def __init__(self):
        super()._init_()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fusion=nn.Sequential(
            nn.Linear(16 + 6, 32),
            nn.ReLU(),
        )
    def forward(self,depth,state):
        image_depth=encoder(depth)
        fuse=torch.cat([image_depth,state],dim=1)
        return self.fusion(fuse)
depth=torch.randn(2,1,80,120)
state=torch.randn(2,6)
model=DepthStateFusion()
feature=model(depth,state)
print(image_depth.shape,fuse.shape,feature.shape)
