# Linear Regression

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
num_inputs = 2 # dimension of featurs
num_examples = 1000 # instances
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
# print(features)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)# input noise with Normal Distribution mean=0 ,standard deviation =0.01

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
    return 0
# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# # from d2lzh_pytorch import *
#
# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

# 本函数已保存在d2lzh包中方便以后使用
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)  # 样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
#         yield  features.index_select(0, j), labels.index_select(0, j) #index_select 0means select row


# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     # print(X.shape)
#     # print(y.shape)
#     break
# Initialization of training weights
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32,requires_grad=True)
# b = torch.zeros(1, dtype=torch.float32,requires_grad=True)

# need back propagation to require gradient
# w.requires_grad_(requires_grad=True)
# b.requi_grad_(requires_grad=True)






## Using Neural Network Linear to create LinearRegression
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)

# for X,y in data_iter:
#     print(X,y)

# Up to now we create a dataset with feature and label *1000


import torch.nn as nn
class LinearRegression(nn.Module):
    def __init__(self,n_features):
        super(LinearRegression,self).__init__()
        self.linear1 = nn.Linear(n_features,100)
        self.linear2 = nn.Linear(100,1)
    def forward(self,x):
        x = self.linear1(x)
        y = self.linear2(x)
        return y

Linearnet = LinearRegression(num_inputs)

print(Linearnet)



# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net2 = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

# print(net)
# print(net2)
# print(net3)

from torch.nn import init
init.normal_(Linearnet.linear1.weight,mean=0,std=0.01)
init.constant_(Linearnet.linear1.bias,val=0)

# for parameters in Linearnet.parameters():
#     print(parameters)

# loss function
loss = nn.MSELoss()

#optimizer
import torch.optim as optim
optimizer = optim.Adagrad([
    {'params': Linearnet.linear1.parameters(),'lr_decay':0.00001},
    {'params': Linearnet.linear2.parameters(),'lr':0.0003}
],lr=0.01,lr_decay=0.0000001)
print(optimizer)
for param in optimizer.param_groups:
    param['lr'] *= 10
print(optimizer)

num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output = Linearnet(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print(output)
print(y)
weightdense=Linearnet.linear1
print(true_w,weightdense.weight)
print(true_b,weightdense.bias)