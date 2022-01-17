import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    def __init__(self):
        # 定义神经网络结构
        super(NN, self).__init__()
        # 第一层 卷积层 输入频道1 输出6 卷积3*3
        self.conv1 = nn.Conv2d(3, 6, 3)
        # 第二层 卷积层
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 第三层 全连接层
        self.fc1 = nn.Linear(16 * 28 * 28, 512)
        # 第四层 全连接层
        self.fc2 = nn.Linear(512, 64)
        # 第五层 全连接层
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 16 * 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x