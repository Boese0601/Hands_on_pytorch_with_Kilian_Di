import time
import torch
from torch import nn, optim
import torchvision.models
import numpy as np
import matplotlib.pylab as plt
from IPython import display
from torch.nn import init
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
sys.path.append('./run.py')
from run import load_data_fashion_mnist,train_gpu

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]) # x = batch_size * C * W * H


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv: # use conv 1*1 to change the  amount of the channel so that X,Y have the same shape to add.
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)



class ResNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,first_block=False):
        super(ResNetBlock,self).__init__()
        if first_block:
            assert in_channels == out_channels
            self.conv1 = ResidualBlock(in_channels=in_channels,out_channels=in_channels)
            self.conv2 = ResidualBlock(in_channels=in_channels,out_channels=in_channels)
        else:
            self.conv1 = ResidualBlock(in_channels=in_channels,out_channels=out_channels,use_1x1conv=True,stride=2)
            self.conv2 = ResidualBlock(in_channels=out_channels,out_channels=out_channels)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.Initialnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_block1 = ResNetBlock(64, 64,first_block=True)
        self.resnet_block2 = ResNetBlock(64, 128)
        self.resnet_block3 = ResNetBlock(128,256)
        self.resnet_block4 = ResNetBlock(256,512)
        self.GlobalAvgpooling = GlobalAvgPool2d()
        self.linear = nn.Linear(512,10)

    def forward(self,x):
        x = self.Initialnet(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.GlobalAvgpooling(x)
        x = self.linear(x.view(x.shape[0],-1))
        return x

if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    CDNet = ResNet18()
    X = torch.randn(1,1,224,224)
    print(CDNet)
    batch_size = 32
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.001, 2
    optimizer = torch.optim.Adam(CDNet.parameters(), lr=lr)
    train_gpu(net=CDNet, train_iter=train_iter, test_iter=test_iter, batch_size=batch_size, optimizer=optimizer,
              device=device, num_epochs=num_epochs)
    torch.save(CDNet.state_dict(), './ResNet18.pth')