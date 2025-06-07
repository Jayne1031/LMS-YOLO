import torch
import torch.nn as nn


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


class LCA(nn.Module):
    def __init__(self, in_channels, rate=4, k_sizes=[3, 5], dilations=[1, 2], rank=2, groups=2):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels // rate)
        self.groups = groups
        # 简化通道注意力部分，仅使用全局平均池化和一个 1x1 卷积层
        self.channel_att_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // self.groups, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.dw_conv_spatial_h_list = nn.ModuleList()
        self.pw_conv_spatial_h_list = nn.ModuleList()
        self.dw_conv_spatial_v_list = nn.ModuleList()
        self.pw_conv_spatial_v_list = nn.ModuleList()
        for k_size, dilation in zip(k_sizes, dilations):
            if k_size == 3:
                k_h, pad_h = (1, 1), (0, 0)
                k_v, pad_v = (1, 1), (0, 0)
            elif k_size == 5:
                k_h, pad_h = (1, 2), (0, 1)
                k_v, pad_v = (2, 1), (1, 0)
            else:
                k_h, pad_h = (1, 2), (0, 1)
                k_v, pad_v = (2, 1), (1, 0)
            # output_channel = in_channels // self.groups
            self.dw_conv_spatial_h_list.append(nn.Conv2d(in_channels // self.groups, in_channels , kernel_size=k_h, stride=1, padding=pad_h, groups=in_channels // self.groups, dilation=dilation))
            self.pw_conv_spatial_h_list.append(LowRankPointwiseConv2d(in_channels, in_channels// self.groups  , rank))
            self.dw_conv_spatial_v_list.append(nn.Conv2d(in_channels// self.groups, in_channels , kernel_size=k_v, stride=1, padding=pad_v, groups=in_channels // self.groups, dilation=dilation))
            self.pw_conv_spatial_v_list.append(LowRankPointwiseConv2d(in_channels, in_channels// self.groups, rank))
        # 去除可变形卷积
        self.avg_pool_list = nn.ModuleList([nn.AvgPool2d(kernel_size=pool_size, stride=stride, padding=padding) for pool_size,stride, padding in zip([3, 5],[1,1],[1,2])])
        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels// self.groups, in_channels// self.groups, kernel_size=3, padding=1) for _ in range(len(k_sizes))])
        self.group_conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups)
        # 使用 InstanceNorm2d 代替 LayerNorm
        self.group_norm = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels * (self.groups+1), inchannel_rate, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu_spatial = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_conv1x1(x)
        x = self.group_norm(x)
        group_channels = c // self.groups
        group_x = x.reshape(b, self.groups, group_channels, h, w)
        # 计算通道注意力
        x_channel_att = self.channel_att_conv(x)
        # import pdb
        # pdb.set_trace()
        x_ca = group_x * x_channel_att
        attn_v_list = []
        for i in range(self.groups):
            attn_v = x_ca[:, i]
            for dw_h, pw_h, dw_v, pw_v in zip(self.dw_conv_spatial_h_list, self.pw_conv_spatial_h_list, self.dw_conv_spatial_v_list, self.pw_conv_spatial_v_list):
                # import pdb
                # pdb.set_trace()
                attn_h = dw_h(attn_v)
                attn_h = pw_h(attn_h)
                attn_v = dw_v(attn_h)
                attn_v = pw_v(attn_v)
            pooled_feats = []

            for pool, conv in zip(self.avg_pool_list, self.conv_list):
                pooled = pool(attn_v)
                pooled = conv(pooled)
                pooled_feats.append(pooled)

            attn_v = torch.cat([attn_v] + pooled_feats, dim=1)
            attn_v_list.append(attn_v)

        attn_v = torch.stack(attn_v_list, dim=1)
        attn_v = attn_v.view(b, -1, h, w)
        attn_low = self.relu_spatial(self.norm1(self.conv1(attn_v)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(attn_low)))
        # 特征融合，使用加权求和
        # import pdb
        # pdb.set_trace()
        x_ca = x_ca.view(b, c, h, w)
        out = x_ca * 0.6 + x_spatial_att * 0.4
        out = out.view(b, c, h, w)
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

class C2f_LCA2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.mgam = LCA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.mgam(out)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.mgam(out)


if __name__ == "__main__":
    test_input = torch.randn(2, 64, 32, 32)
    model = LCA(in_channels=64)
    with torch.no_grad():
        out = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {out.shape}")