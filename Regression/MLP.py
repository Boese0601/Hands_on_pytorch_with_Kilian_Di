# Multi-Layer-Perceptron
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pylab as plt
from IPython import display
from torch.nn import init
import torch.nn as nn
# from Day2 import set_figsize
# from Day3 import load_data_fashion_mnist

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def load_data_fashion_mnist(batch_size):
    num_workers=4
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                 transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter
# plot relu
def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()

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


# example of relu
# x = torch.arange(-10,10,0.1,requires_grad=True)
# y = x.relu()
# xyplot(x,y,'relu')


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# print(train_iter,test_iter)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        y = x.view(x.shape[0],-1)
        return y

class MLP(nn.Module):
    def __init__(self,num_inputs=None,num_hidden=None,num_outputs=None):
        super(MLP,self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.flatten = Flatten()
        self.linear1 = nn.Linear(self.num_inputs,self.num_hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        # self.batchnorm = nn.BatchNorm1D()
        self.linear2 = nn.Linear(self.num_hidden,self.num_outputs)

    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        output = self.linear2(x)
        return output

netmlp = MLP(num_inputs=num_inputs, num_outputs=num_outputs, num_hidden=num_hiddens)
print(netmlp)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(netmlp.parameters(), lr=0.5)

num_epochs = 5


# training process
train_ch3(netmlp, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# Testing process
netmlp.eval()
X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(netmlp(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
PATH = './MLP2.pth'
torch.save(netmlp.state_dict(),PATH)