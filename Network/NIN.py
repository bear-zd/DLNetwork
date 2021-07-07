import torch
import torch.nn as nn

def MLPBlock(in_channels,out_channels,ksize,stride,pad):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=ksize,stride=stride,padding=pad),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
        nn.ReLU())
    return block

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class NIN(nn.Module):
    def __init__(self,in_channels,class_num):
       self.net = nn.Sequential(
            MLPBlock(in_channels,96,11,4,0),
            nn.MaxPool2d(kernel_size=3,stride=2),
            MLPBlock(96,256,5,1,2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            MLPBlock(256,384,3,1,1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Dropout(),
            MLPBlock(384,class_num,3,1,1),
            nn.AdaptiveAvgPool2d(),
            FlattenLayer()
        )
    def forward(self,x):
        x = self.net(x)
        return x

