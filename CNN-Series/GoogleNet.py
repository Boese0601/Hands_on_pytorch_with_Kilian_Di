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

class InceptionBlock(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(InceptionBlock,self).__init__()
        self.channel1 = nn.Conv2d(in_channels=in_c,out_channels=c1,kernel_size=1)
        self.relu1 = nn.ReLU()


        self.channel2_1 = nn.Conv2d(in_channels=in_c,out_channels=c2[0],kernel_size=1)
        self.relu2 = nn.ReLU()
        self.channel2_2 = nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()


        self.channel3_1 = nn.Conv2d(in_channels=in_c,out_channels=c3[0],kernel_size=1)
        self.relu4 = nn.ReLU()
        self.channel3_2 = nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5,padding=2)
        self.relu5 = nn.ReLU()


        self.channel4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.channel4_2 = nn.Conv2d(in_channels=in_c,out_channels=c4,kernel_size=1)
        self.relu6 = nn.ReLU()


    def forward(self,x):
        x1 = self.relu1(self.channel1(x))
        x2 = self.relu3(self.channel2_2(self.relu2(self.channel2_1(x))))
        x3 = self.relu5(self.channel3_2(self.relu4(self.channel3_1(x))))
        x4 = self.relu6(self.channel4_2(self.channel4_1(x)))
        output = torch.cat((x1,x2,x3,x4),dim=1)


        return  output

class b1(nn.Module):
    def __init__(self):
        super(b1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x = self.pool(self.relu1(self.conv1(x)))
        return x


class b2(nn.Module):
    def __init__(self):
        super(b2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x = self.pool(self.conv2(self.conv1(x)))
        return x

class b3(nn.Module):
    def __init__(self):
        super(b3,self).__init__()
        self.Inception1 = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.Inception2 = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.pool = nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x = self.pool(self.Inception2(self.Inception1(x)))
        return x

class b4(nn.Module):
    def __init__(self):
        super(b4,self).__init__()
        self.Inception1 = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.Inception2 = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.Inception3 = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.Inception4 = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.Inception5 = InceptionBlock(528, 256, (160, 320), (32, 128), 128)
        self.pool = nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x = self.pool(self.Inception5(self.Inception4(self.Inception3(self.Inception2(self.Inception1(x))))))
        return x

class b5(nn.Module):
    def __init__(self):
        super(b5,self).__init__()
        self.Inception1 = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.Inception2 = InceptionBlock(832, 384, (192, 384), (48, 128), 128)
        self.GlobalPool = GlobalAvgPool2d()
    def forward(self,x):
        return self.GlobalPool(self.Inception2(self.Inception1(x)))

class GooleNet(nn.Module):
    def __init__(self):
        super(GooleNet,self).__init__()
        self.b1 = b1()
        self.b2 = b2()
        self.b3 = b3()
        self.b4 = b4()
        self.b5 = b5()
        self.linear = nn.Linear(1024,10)

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        out = self.b5(x)

        return self.linear(out.view(out.shape[0],-1))

if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    Inception = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
    X = torch.rand(1, 1, 96, 96)
    net = GooleNet()
    print(net)
    print(net(X))
    batch_size = 1
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.001, 2
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_gpu(net=net, train_iter=train_iter, test_iter=test_iter, batch_size=batch_size, optimizer=optimizer,
              device=device, num_epochs=num_epochs)
    torch.save(net.state_dict(), './VGG11.pth')
