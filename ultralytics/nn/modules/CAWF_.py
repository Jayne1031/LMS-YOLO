import math

import numpy as np
import torch
import torch.nn as nn



# 三个分支concat操作
class CAWF(nn.Module):
    def __init__(self, dimension=1):
        super(CAWF, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)
