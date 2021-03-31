import torch
from torch import nn
from torch.nn import init

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层


    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
# net = MLP()
# print(net)
# net(X)
from collections import OrderedDict

class MySequential(nn.Module):
# OrderedDict([
#         ('name of layer',nn.LayerType()),
#         ………………
#         ('linear',nn.Linear(numinputs,numoutputs))
#               ])
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        )#Eual to Sequential
# print(net)
# net(X)

# Using List To create NN
# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10)) # # 类似List的append操作
# print(net[-1])  # 类似List的索引访问
# print(net)
# Implement of ModuleList
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
#!!!
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) +(x)
        return x

# ModuleDict similar to ModuleList
# net = nn.ModuleDict({
#     'linear': nn.Linear(784, 256),
#     'act': nn.ReLU(),
# })
# net['output'] = nn.Linear(256, 10) # 添加
# print(net['linear']) # 访问
# print(net.output)
# print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError

net =MLP()
# # print(net)
# for name,para in net.named_parameters():# Not Optimizer,But the Neural Network
#     print(name,para.shape)


# Individual Layer
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()

print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(y)
print(y.mean().item())
