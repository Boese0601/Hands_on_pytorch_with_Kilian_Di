import torch
import numpy
# a = torch.arange(1,5,1)
# b = torch.linspace(1,4,4)
# print(a,b)
# x = torch.rand(5,3)
# y = torch.randn_like(x)
# print(x)
# print(y)
# print(x.size())
# print(y.shape)
# print(x.reshape(-1,15))
# print(x)
# print(x+y)
# a=x[0][0]
# print(a)
# print(a.item())
# a1 = torch.arange(1,5,1)
# a2 = torch.arange(2,8,1).view(-1,1)
# print(a1+a2)
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# torch.add(x, y, out=y)
# print(y)
# print(id(y)==id_before)
# a = y.numpy()
# print(a)
# b= numpy.linspace(1,5,2)
#
# print(torch.from_numpy(b))
# 以下代码只有在PyTorch GPU版本上才会执行
# y = torch.arange(1,11,1,requires_grad=True,dtype=float).view(-1,5)
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # GPU
#     y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
#     x = x.to(device)                       # 等价于 .to("cuda")
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
# print(x.grad_fn)
# print(x)

# x = y**2
# # print(y)
# # print(y.grad_fn)
# # a = y.mean()
# # print(a)
# # print(a.grad_fn).
# # 再来反向传播一次，注意grad是累加的
# print(x)
# out2 = x.sum()
# out2.backward()
# print(x.grad)

# out3 = x.sum()
# # x.grad.data.zero_()
# out3.backward()
# print(x.grad)
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# y = 2 * x
# z = y.view(2, 2)
# print(z)
# v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
# z.backward(v)
# print(x.grad)
y2 = x**2
z2 = y2.view(2,2)
v2 =torch.ones(1,4,dtype=torch.float).view(-1,2)
print(v2)
z2.backward(v2)
print(z2)
print(x.grad)