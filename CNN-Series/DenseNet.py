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
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]) # x = batch_size * C * W * H

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)



if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlosk_%d" % i, DB)
        # 上一个稠密块的输出通道数
        num_channels = DB.out_channels
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))

    X = torch.rand((1, 1, 96, 96))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape:\t', X.shape)

    CDNet = net

    print(CDNet)
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(CDNet.parameters(), lr=lr)
    train_gpu(net=CDNet, train_iter=train_iter, test_iter=test_iter, batch_size=batch_size, optimizer=optimizer,
              device=device, num_epochs=num_epochs)
    torch.save(CDNet.state_dict(), './DenseNet.pth')