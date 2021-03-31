import time
import torch
from torch import nn, optim
import torchvision.models
import numpy as np
import matplotlib.pylab as plt
from IPython import display
from torch.nn import init
import torchvision.transforms as transforms



def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


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



# loading Data with small batch

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[]
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform =torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                 transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter


def evaluate_accuracy_cpu(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def evaluate_accuracy_gpu(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():# turn off gradient tracking Because of evaluation mode
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
# train with cpu
def train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
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
        test_acc = evaluate_accuracy_cpu(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    return y_hat

def train_gpu(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_gpu(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

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