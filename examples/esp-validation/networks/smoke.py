"""Small input programs to audit the clean compiler's current coverage."""
import torch
from torch import nn

class Gemm(nn.Module):
    def forward(self, a, b):
        return a @ b

class Add(nn.Module):
    def forward(self, a, b):
        return torch.relu(a+b)

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,5,3,padding=1)
    def forward(self,x):
        return torch.relu(self.conv(x))

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Conv2d(3,4,3,padding=1)
        self.second=nn.Conv2d(4,4,3,padding=1)
        self.shortcut=nn.Conv2d(3,4,1)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(4,7)
    def forward(self,x):
        y=self.second(torch.relu(self.first(x)))
        y=torch.relu(y+self.shortcut(x))
        return self.classifier(self.pool(y).flatten(1))

def create_model(case='gemm'):
    torch.manual_seed(19)
    if case=='gemm':return Gemm(),(torch.randn(5,3),torch.randn(3,7))
    if case=='add':return Add(),(torch.randn(1,3,5,7),torch.randn(1,3,5,7))
    if case=='conv':return Conv(),(torch.randn(1,3,5,7),)
    if case=='residual':return Residual(),(torch.randn(1,3,5,7),)
    raise ValueError(case)
