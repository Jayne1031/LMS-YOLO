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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, rank=1, groups=1):
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
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class LCA(nn.Module):
    def __init__(self, in_channels, rate=4, k_sizes=[3, 5], dilations=[2, 2], rank=1, groups=4):  # 将 groups 修改为 4
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels // rate)
        self.k_sizes = k_sizes
        self.groups = groups
        self.channel_att_conv = ECA(in_channels)
        self.conv_spatial_h_list = nn.ModuleList()
        self.conv_spatial_v_list = nn.ModuleList()
        self.dilated_conv_spatial_h_list = nn.ModuleList()
        self.dilated_conv_spatial_v_list = nn.ModuleList()

        group_in_channels = in_channels // groups
        group_out_channels = group_in_channels

        # for group in range(groups):
        #     # 为不同组分配不同的 k_size 和 dilation
        #     group_k_sizes = [(k_size + group) % len(k_sizes) for k_size in k_sizes]
        #     group_dilations = [(dilation + group) % len(dilations) for dilation in dilations]
        #     # import pdb
        #     # pdb.set_trace()
        #     for k_size, dilation in zip(self.k_sizes, group_dilations):
        #         if k_size == 0:  # 处理可能的 0 值
        #             k_size = len(k_sizes)
        #         if dilation == 0:  # 处理可能的 0 值
        #             dilation = len(dilations)
        #
        #         if k_size == 3:
        #             k_h, pad_h = (1, 2), (0, 1)
        #             k_v, pad_v = (2, 1), (1, 0)
        #         elif k_size == 5:
        #             k_h, pad_h = (1, 5), (0, 2)
        #             k_v, pad_v = (5, 1), (2, 0)
        #         # else:
        #         #     k_h, pad_h = (1, 3), (0, 3)
        #         #     k_v, pad_v = (3, 1), (3, 0)
        #
        #         # 普通卷积
        #         self.conv_spatial_h_list.append(
        #             SeparableLowRankConv2d(
        #                 group_in_channels, group_out_channels, kernel_size=k_h, stride=1, padding=pad_h,
        #                 dilation=1, rank=rank, groups=1
        #             )
        #         )
        #         self.conv_spatial_v_list.append(
        #             SeparableLowRankConv2d(
        #                 group_in_channels, group_out_channels, kernel_size=k_v, stride=1, padding=pad_v,
        #                 dilation=1, rank=rank, groups=1
        #             )
        #         )
        #
        #         # 膨胀卷积
        #         self.dilated_conv_spatial_h_list.append(
        #             SeparableLowRankConv2d(
        #                 group_in_channels, group_out_channels, kernel_size=k_h, stride=1, padding=pad_h,
        #                 dilation=2, rank=rank, groups=1
        #             )
        #         )
        #         self.dilated_conv_spatial_v_list.append(
        #             SeparableLowRankConv2d(
        #                 group_in_channels, group_out_channels, kernel_size=k_v, stride=1, padding=pad_v,
        #                 dilation=2, rank=rank, groups=1
        #             )
        #         )
        self.conv0h_list = nn.ModuleList()
        self.conv0v_list = nn.ModuleList()
        self.conv_spatial_h_list = nn.ModuleList()
        self.conv_spatial_v_list = nn.ModuleList()
        for k_size in k_sizes:
            if k_size == 3:
                self.conv0h_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels,
                    out_channels=group_out_channels,
                    kernel_size=(1, 2),
                    stride=1,
                    rank=rank,
                    padding=(0, 1)
                ))
                self.conv0v_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels,
                    out_channels=group_out_channels,
                    kernel_size=(2, 1),
                    stride=1,
                    rank=rank,
                    padding=(1, 0)
                ))
                self.conv_spatial_h_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels,
                    out_channels=group_out_channels,
                    kernel_size=(1, 1),
                    rank=rank,
                    stride=1,
                    padding=(0, 0),
                    dilation=2
                ))
                self.conv_spatial_v_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels,
                    out_channels=group_out_channels,
                    kernel_size=(1, 1),
                    rank=rank,
                    stride=1,
                    padding=(0, 0),
                    dilation=2
                ))
            elif k_size == 5:
                self.conv0h_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels, out_channels=group_out_channels,
                    kernel_size=(1, 3), rank=rank, padding=(0, 2),stride=1,
                ))
                self.conv0v_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels, out_channels=group_out_channels,
                    kernel_size=(3, 1), rank=rank, padding=(2, 0),stride=1,
                ))
                self.conv_spatial_h_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels, out_channels=group_out_channels,
                    kernel_size=(1, 3), rank=rank,
                    padding=(0, 1), dilation=2,stride=1,
                ))
                self.conv_spatial_v_list.append(SeparableLowRankConv2d(
                    in_channels=group_in_channels, out_channels=group_out_channels,
                    kernel_size=(3, 1), rank=rank,
                    padding=(1, 0), dilation=2,stride=1,
                ))
        self.avg_pool_list = nn.ModuleList([nn.AvgPool2d(kernel_size=pool_size) for pool_size in [2, 4]])
        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(len(k_sizes))])
        self.group_conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups)
        self.group_norm = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels * (len(self.avg_pool_list) + 1), inchannel_rate, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu_spatial = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.group_conv1x1(x)
        x = self.group_norm(x)
        group_channels = c // self.groups
        group_x = x.reshape(b, self.groups, group_channels, h, w)

        # 存储不同分组的结果
        group_results = []
        for group in range(self.groups):
            group_attn = group_x[:, group]
            # import pdb
            # pdb.set_trace()
            for conv_h, conv_v, dilated_h, dilated_v in zip(
                self.conv0h_list,
                self.conv0v_list,
                self.conv_spatial_h_list,
                self.conv_spatial_v_list
            ):
                attn_h = conv_h(group_attn)
                attn_v = conv_v(attn_h)
                attn_h = dilated_h(attn_v)
                attn_v = dilated_v(attn_h)
            group_results.append(attn_v)

        attn_v = torch.cat(group_results, dim=1)
        pooled_feats = []
        for pool, conv in zip(self.avg_pool_list, self.conv_list):
            pooled = pool(attn_v)
            pooled = conv(pooled)
            pooled_feats.append(nn.functional.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False))
        # import pdb
        # pdb.set_trace()
        attn_v = torch.cat([attn_v] + pooled_feats, dim=1)
        attn_low = self.relu_spatial(self.norm1(self.conv1(attn_v)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(attn_low)))

        # 计算通道注意力
        x_channel_att = self.channel_att_conv(x)
        x_ca = x * x_channel_att

        # 特征融合
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

class C2f_LCA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.mgam = LCA(c2,groups=4)

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
# 测试代码
if __name__ == "__main__":
    test_input = torch.randn(2, 64, 32, 32)
    lca_model = LCA(in_channels=64, groups=8)  # 创建 LCA 模型时设置 groups 为 4
    with torch.no_grad():
        lca_out = lca_model(test_input)
    print(f"LCA Input shape: {test_input.shape}")
    print(f"LCA Output shape: {lca_out.shape}")