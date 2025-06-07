import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankPointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, rank):
        super(LowRankPointwiseConv2d, self).__init__()
        self.rank = rank
        self.conv_reduce = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.conv_expand = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.conv_expand(x)
        return x


class SeparableLowRankConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, rank=1):
        super(SeparableLowRankConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = LowRankPointwiseConv2d(in_channels, out_channels, rank)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        # 转换维度为 [B, C, 1] 后做 1D 卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)  # [B, C, 1, 1]
        return x * y


class LCA_layers(nn.Module):
    def __init__(self, in_channels, rate=4, k_size=7):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)
        self.k_size = k_size

        # -- 修正1: 定义激活函数 (ReLU 或其他) --
        self.act = nn.ReLU(inplace=True)

        # Spatial attention layers (rank=1，深度可分离+低秩)
        self.conv0h = SeparableLowRankConv2d(
            in_channels, in_channels,
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), rank=1
        )
        self.conv0v = SeparableLowRankConv2d(
            in_channels, in_channels,
            kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), rank=1
        )
        self.conv_spatial_h = SeparableLowRankConv2d(
            in_channels, in_channels,
            kernel_size=(1, 3), stride=(1, 1), padding=(0, 2),
            dilation=2, rank=1
        )
        self.conv_spatial_v = SeparableLowRankConv2d(
            in_channels, in_channels,
            kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),
            dilation=2, rank=1
        )

        # 通道维度减缩后再扩张
        self.conv1 = LowRankPointwiseConv2d(in_channels, inchannel_rate, rank=1)
        self.conv2 = LowRankPointwiseConv2d(inchannel_rate, out_channels, rank=1)

        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        # 通道注意力 (ECA)
        self.eca = ECA(channel=in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # x2 = x.clone()
        # 1) Channel attention
        x_channel_att = self.eca(x)
        x = x * x_channel_att  # 融合到主分支

        # 2) Spatial attention
        # 复制一份融合后的 x
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)

        # -- 修正2: 用 self.act 而非 self.relu --
        attn = self.act(self.norm1(attn))
        attn = self.sigmoid(self.norm2(self.conv2(attn)))

        # 3) 最终输出：通道注意力后的特征再乘上空间注意力
        # -- 修正3: 移除多余的 '*' 并正确拼接 --
        out = u * attn
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

class C2f_LCA3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.mgam = LCA_layers(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        out = self.mgam(out)
        # out = out + out1
        return out

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.mgam(out)

# 测试代码
if __name__ == "__main__":
    test_input = torch.randn(2, 64, 32, 32)
    lca_model = LCA(in_channels=64)
    with torch.no_grad():
        lca_out = lca_model(test_input)
    print(f"LCA Input shape: {test_input.shape}")
    print(f"LCA Output shape: {lca_out.shape}")
