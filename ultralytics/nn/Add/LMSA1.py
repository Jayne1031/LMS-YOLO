import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankPointwiseConv2d(nn.Module):
    """
    对 1x1 卷积做秩-r 分解:
    W ∈ R^{C_out x C_in} ~ U_r * V_r^T
    """
    def __init__(self, in_channels, out_channels, rank):
        super(LowRankPointwiseConv2d, self).__init__()
        self.rank = rank
        # 先降维
        self.conv_reduce = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        # 再升维
        self.conv_expand = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.conv_expand(x)
        return x


class SeparableLowRankConv2d(nn.Module):
    """
    大核卷积: Depthwise + (Low-rank) Pointwise
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank,
                 stride=1, padding=0, dilation=1):
        super(SeparableLowRankConv2d, self).__init__()
        # Depthwise
        self.dw = nn.Conv2d(in_channels, in_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=in_channels,
                         bias=False)
        # Pointwise (低秩)
        self.pw_lr = LowRankPointwiseConv2d(in_channels, out_channels, rank=rank)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw_lr(x)
        return x

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y
class LMSA(nn.Module):
    def __init__(self,
                 channels,
                 c2=None,        # 保留与原版一致的接口
                 factor=16,      # 分组数
                 k_sizes=[7, 11],       # 减少大核大小列表元素
                 rank=1):        # 进一步降低 Low-rank 分解的秩
        super(LMSA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 仅保留水平池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        group_channels = channels // self.groups
        # 使用 Layer Normalization 代替 GroupNorm
        # self.gn = nn.LayerNorm([group_channels,32,32])
        # 原来的 conv1x1 用可选低秩
        self.conv1x1 = LowRankPointwiseConv2d(
            in_channels=group_channels,  # cat(x_h)
            out_channels=group_channels,
            rank=rank
        )
        # 原来的 3x3 -> 可分离 + 低秩
        self.conv3x3 = SeparableLowRankConv2d(
            in_channels=group_channels,
            out_channels=group_channels,
            kernel_size=3,
            rank=rank,
            padding=1
        )
        # 存储不同分组的 k_size
        self.k_sizes = k_sizes
        # 多尺度大核分支 (k_size=7, 11)
        self.conv0h_list = nn.ModuleList()
        self.conv0v_list = nn.ModuleList()
        self.conv_spatial_h_list = nn.ModuleList()
        self.conv_spatial_v_list = nn.ModuleList()
        for k_size in k_sizes:
            if k_size == 7:
                self.conv0h_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels,
                    out_channels=group_channels,
                    kernel_size=(1, 3),
                    rank=rank,
                    padding=(0, 1)
                ))
                self.conv0v_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels,
                    out_channels=group_channels,
                    kernel_size=(3, 1),
                    rank=rank,
                    padding=(1, 0)
                ))
                self.conv_spatial_h_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels,
                    out_channels=group_channels,
                    kernel_size=(1, 3),
                    rank=rank,
                    padding=(0, 3),
                    dilation=2
                ))
                self.conv_spatial_v_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels,
                    out_channels=group_channels,
                    kernel_size=(3, 1),
                    rank=rank,
                    padding=(3, 0),
                    dilation=2
                ))
            elif k_size == 11:
                self.conv0h_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels, out_channels=group_channels,
                    kernel_size=(1, 3), rank=rank, padding=(0,1)
                ))
                self.conv0v_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels, out_channels=group_channels,
                    kernel_size=(3, 1), rank=rank, padding=(1,0)
                ))
                self.conv_spatial_h_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels, out_channels=group_channels,
                    kernel_size=(1, 5), rank=rank,
                    padding=(0, 5), dilation=2
                ))
                self.conv_spatial_v_list.append(SeparableLowRankConv2d(
                    in_channels=group_channels, out_channels=group_channels,
                    kernel_size=(5, 1), rank=rank,
                    padding=(5, 0), dilation=2
                ))
        # 原来的 conv1 用可分离 or 低秩 1x1
        self.conv1 = LowRankPointwiseConv2d(
            in_channels=group_channels,
            out_channels=group_channels,
            rank=rank
        )
        # 使用 SE 注意力模块，使用更轻量级的激活函数 LeakyReLU
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(group_channels, group_channels // 4, kernel_size=1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(group_channels // 4, group_channels, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # 使用 ECA 模块代替 SE 模块
        self.se = ECA(group_channels)
        self.upconv = nn.Conv2d(group_channels, group_channels, kernel_size=(3,3), stride=(1,1), padding=(0,0))
    def forward(self, x):
        b, c, h, w = x.shape
        # 分组
        # gn = nn.LayerNorm([c // self.groups, h, w])
        group_x = x.reshape(b*self.groups, -1, h, w)
        # 存储不同分组的结果
        group_results = []
        for i in range(self.groups):
            # 根据分组索引选择 k_size
            k_size_index = i % len(self.k_sizes)
            k_size = self.k_sizes[k_size_index]
            # 多尺度卷积

            attn = self.conv0h_list[k_size_index](group_x[i])
            attn = self.conv0v_list[k_size_index](attn)
            attn = self.conv_spatial_h_list[k_size_index](attn)
            attn = self.conv_spatial_v_list[k_size_index](attn)
            attn = self.conv1(attn)
            attn = self.upconv(attn.unsqueeze(0)).squeeze(0)
            group_results.append(attn)
        attn = torch.stack(group_results, dim=0)

        # 水平池化
        x_h = self.pool_h(group_x)  # [b*g, c//g, h, 1]

        # 拼接 + 1x1(低秩) 融合
        hw_cat = x_h   # [b*g, c//g, h, 1]
        hw_fused = self.conv1x1(hw_cat)         # [b*g, c//g, h, 1]
        x_h = hw_fused           # [b*g, c//g, h, 1]


        # 自适应增强分支
        # group_x= group_x.cuda()
        # x_h = x_h.cuda()
        # try:
        #     x1 = gn(group_x * x_h)
        # except:
        #     group_x= group_x.cuda()
        #     x_h = x_h.cuda()
        #     print(group_x.device)
        #     print(x_h.device)
        #     x1 = gn(group_x * x_h)
        x1 = group_x * x_h
        x2 = self.conv3x3(group_x)  # 可分离+低秩 3x3
        # 使用 SE 注意力
        try:
            x1 = x1 * self.se(x1)
        except:
            import pdb
            pdb.set_trace()
        x2 = x2 * self.se(x2)
        # 注意力加权
        x11 = self.softmax(self.agp((attn.unsqueeze(0) * (x1.view(b,-1, c//self.groups ,h,w))).view(-1,c//self.groups ,h,w)).view(b*self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.view(b*self.groups, -1, h*w)
        # import pdb
        # pdb.set_trace()
        x21 = self.softmax(self.agp(
            (attn.unsqueeze(0) * (x2.view(b, -1, c // self.groups, h, w))).view(-1, c // self.groups, h, w)).view(
            b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.view(b * self.groups, -1, h * w)
        # x21 = self.softmax(self.agp(attn * x2).view(b*self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.view(b*self.groups, -1, h*w)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b*self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).view(b, c, h, w)
        return out

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f_LMSA(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.emlsattention = LMSA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.emlsattention(out)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.emlsattention(out)

# 测试代码
if __name__ == "__main__":
    test_input = torch.randn(2, 512, 32, 32)  # batch=2, channel=64, h=32, w=32
    model = LMSA(channels=512)
    with torch.no_grad():
        out = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {out.shape}")