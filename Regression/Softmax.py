# Softmax Regression Multi-Class Regression
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from IPython import display
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from torch.nn import init

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
# print(feature)
# print(feature.shape, label)  # Channel x Height x Width
# print(feature.view(feature.shape[0],-1))

# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# loading Data with small batch
batch_size = 256
num_workers = 4 # Multi-processing
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                 transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# SOftmax Regression
num_inputs = 784 #Image with (H,W) 28*28
num_outputs = 10 #Image classes

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

# Define By Hand
class SoftMaxNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(SoftMaxNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self,x):
        y = self.linear(x.view(x.shape[0],-1)) # resize to 1 * 784
        return y

net = SoftMaxNet(num_inputs,num_outputs)

#  Define by Sequential
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        y = x.view(x.shape[0],-1)
        return y
# Define by OrderedDict
net2 = nn.Sequential(
    OrderedDict([
        ('flatten',Flatten()),
        ('linear',nn.Linear(num_inputs,num_outputs))
    ])
)
print(net)
print(net2)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.09)
print(optimizer)

num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                optimizer = torch.optim.SGD(net.parameters(),lr=lr,batch_size=batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    return y_hat
train_ch3(net,train_iter,test_iter, loss, num_epochs, batch_size, None, None, optimizer)
X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])

