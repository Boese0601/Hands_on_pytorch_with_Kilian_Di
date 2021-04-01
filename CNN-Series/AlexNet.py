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



class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,2)
        self.conv3 = nn.Conv2d(256,384,3,1,1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3,2)
        self.averagepool =nn.AdaptiveAvgPool2d(output_size=(5,5))
        self.linear1 = nn.Linear(256*5*5,4096)
        self.relu6 =nn.ReLU()
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096,4096)
        self.relu7 =nn.ReLU()
        self.dp2 = nn.Dropout(0.5)
        self.outlayer = nn.Linear(4096,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        # print('before Avg:',x.shape)
        x = self.averagepool(x)
        # print('after Avg:',x.shape)
        x = self.linear1(x.view(x.shape[0],-1))
        x = self.relu6(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.dp2(x)
        output = self.outlayer(x)
        return output

if __name__ ==  '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 128
    num_workers = 4  # Multi-processing
    num_epochs = 12
    net = MyAlexNet()
    print(net)
    train_iter,test_iter = load_data_fashion_mnist(batch_size,resize=224)
    optimizer = optim.Adam(params=net.parameters(),lr=0.001)
    train_gpu(net=net,train_iter=train_iter,test_iter=test_iter,batch_size=batch_size,optimizer=optimizer,device=device,num_epochs=num_epochs)
    torch.save(net.state_dict(),'./AlexNet.pth')
