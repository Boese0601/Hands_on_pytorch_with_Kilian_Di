import time
import torch
from torch import nn, optim
import torchvision.models
import numpy as np
import matplotlib.pylab as plt
from IPython import display
from torch.nn import init
import torchvision.transforms as transforms

import sys
sys.path.append('./run.py')
from run import load_data_fashion_mnist,train_gpu


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk) # blk is a ModuleList,we need to instantiate the Convolution Blocks
# Define VGG Convolution Blocks
class VGGConvblock(nn.Module):
    def __init__(self):
        super(VGGConvblock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,128,3,padding=1)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128,256,3,padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256,256,3,padding=1)
        self.relu4 = nn.ReLU()
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256,512,3,padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512,512,3,padding=1)
        self.relu6 = nn.ReLU()
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512,512,3,padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512,512,3,padding=1)
        self.relu8 = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pooling3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pooling4(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        out = self.pooling(x)
        return out
# Define VGG inear Blocks
class VGGLinear(nn.Module):
    def __init__(self):
        super(VGGLinear,self).__init__()
        self.linear1 = nn.Linear(512*7*7,4096)
        self.relu1 = nn.ReLU()
        self.dp1 =nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096,4096)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.5)
        self.outlayer = nn.Linear(4096,10)

    def forward(self,x):
        x = self.linear1(x.view(x.shape[0],-1))
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dp2(x)
        out = self.outlayer(x)
        return out
# Define VGG-11 Model
class VGG11(nn.Module):

    def __init__(self):
        super(VGG11,self).__init__()
        self.vggconv = VGGConvblock()
        self.vgglinear = VGGLinear()
    def forward(self,x):
        x = self.vggconv(x)
        out = self.vgglinear(x)
        return out
if __name__ ==  '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    MyVGG11 = VGG11()
    print(MyVGG11)
    # Test Output
    # X = torch.randn(1,1,224,224)
    # y = VGG11(X)
    # print(y)
    batch_size = 32
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(MyVGG11.parameters(), lr=lr)
    train_gpu(net=MyVGG11,train_iter=train_iter,test_iter=test_iter,batch_size=batch_size,optimizer=optimizer,device=device,num_epochs=num_epochs)
    torch.save(MyVGG11.state_dict(), './VGG11.pth')
