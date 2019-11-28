import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # 调用父类的初始化
        super(Net,self).__init__()
        # 输入图像通道为1，输出通道为：6 5*5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y=wx+b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] #除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

input  = torch.randn(1, 1, 32, 32)
target = torch.randn(10)
target = target.view(1,-1)

def test_loss():
    net = Net()
    out = net(input)
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print("loss={}".format(loss))
    print(loss.grad_fn)#MSEloss
    print(loss.grad_fn.next_functions[0][0])# liner
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])# relu

    net.zero_grad()
    print("conv1.bias.grad before backward")
    print(net.conv1.bias.grad)
    loss.backward()
    print("conv1.bias.grad after backward")
    print(net.conv1.bias.grad)

import torch.optim as optim

def test_backward():
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    # 在训练的迭代中：进行
    optimizer.zero_grad() #清零梯度缓存
    out = net(input)
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    loss.backward()
    optimizer.step() # 更新参数
    print("loss{}".format(loss))

def test_net():
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    # ---------------
    out = net(input)
    print(out)

if __name__ == "__main__":
    test_backward()


    
