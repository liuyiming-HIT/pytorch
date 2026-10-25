
torch.save(model.state_dict(),"best.pth")
model2=nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)
state=torch.load("best.pth",weight_only=true)
model2.load_state_dict(state)
