import torch.nn as nn
from typing import Optional
import torch

__all__ = ['C2f_UIB']


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    This is to ensure that all layers have channels that are divisible by 8.
    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.
    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

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

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv

class LowRankPointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, rank=1, isact=True):
        super(LowRankPointwiseConv2d, self).__init__()
        self.rank = rank
        self.conv_reduce = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False,padding=0,stride=1)
        self.conv_expand = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False,padding=0,stride=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.isact = isact
        if self.isact:
            self.act = nn.ReLU6()

    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.conv_expand(x)
        x = self.norm(x)
        if self.isact:
            x = self.act(x)
        return x
class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size=3,
                 middle_dw_kernel_size=3,
                 middle_dw_downsample=1,
                 stride=1,
                 expand_ratio=1
                 ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        # self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        self._expand_conv = LowRankPointwiseConv2d(inp, expand_filters, rank=1,isact=True)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        # self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        self._proj_conv = LowRankPointwiseConv2d(expand_filters, oup, rank=1, isact=False)
        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


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

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(UniversalInvertedBottleneckBlock(c_, c_) for _ in range(n)))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        x = x.chunk(2, 1)
        y = list(x)
        # y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2_UIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else UniversalInvertedBottleneckBlock(self.c, self.c) for _ in range(n)
        )

class C3k2_LCA3(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        self.mgam = LCA_layers(c2)

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

class C3k2_LCA3_2(nn.Module):
    """
    改进思路3:
    在C3k2整体完成后，再用LCA处理输出特征。
    通常用于主干(backbone)，对C3k2输出特征做注意力增强。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c_mid = int(c2 * e)
        # 1) split
        self.cv1 = nn.Conv2d(c1, 2*self.c_mid, kernel_size=1, stride=1, bias=False)

        # 2) repeated Bottleneck or C3k
        self.m = nn.ModuleList()
        for _ in range(n):
            if c3k:
                self.m.append(C3k(self.c_mid, self.c_mid, n=2, shortcut=shortcut, g=g))
            else:
                self.m.append(Bottleneck(self.c_mid, self.c_mid, shortcut=shortcut, g=g))

        # 3) 1x1conv 投影
        self.cv2 = nn.Conv2d((2+n)*self.c_mid, c2, kernel_size=1, stride=1, bias=False)

        # 4) LCA
        self.lca = LCA_layers(in_channels=c2)

    def forward(self, x):
        y_split = self.cv1(x).chunk(2, dim=1)  # (split1, split2)
        out_list = list(y_split)

        for block in self.m:
            out_list.append(block(out_list[-1]))

        out = torch.cat(out_list, dim=1)  # [B, 2*c_mid, H, W]
        out = self.cv2(out)               # [B, c2, H, W]

        # 最后插入 LCA
        out = self.lca(out)
        return out


class C3k2_UIB_Final(nn.Module):
    """
    改进思路1:
    在颈部C3k2结构的最终投影阶段使用 UIB 替换原先的 1x1 卷积。
    即: split -> [repeated Bottleneck/C3k] -> concat -> (UIB投影到c2通道).
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 uib_kwargs=None):
        """
        参数:
          c1, c2: 输入/输出通道数
          n: 重复次数
          c3k: 若True则使用C3k替换Bottleneck
          e: 中间通道扩张因子, c_mid=int(c2*e)
          g, shortcut: 同原C3k2
          uib_kwargs: 传入给UniversalInvertedBottleneckBlock的可选字典,
                      如 { 'expand_ratio':1, 'start_dw_kernel_size':3, ...}
        """
        super().__init__()
        if uib_kwargs is None:
            uib_kwargs = {}

        self.c_mid = int(c2 * e)
        # 1. split
        self.cv1 = nn.Conv2d(c1, 2*self.c_mid, kernel_size=1, stride=1, bias=False)

        # 2. repeated子模块: Bottleneck or C3k
        self.m = nn.ModuleList()
        for _ in range(n):
            if c3k:
                self.m.append(C3k(self.c_mid, self.c_mid, n=2, shortcut=shortcut, g=g))
            else:
                self.m.append(Bottleneck(self.c_mid, self.c_mid, shortcut=shortcut, g=g))

        # 3. 使用UIB做最后的投影(2*c_mid -> c2)
        # from universal_inverted_bottleneck import UniversalInvertedBottleneckBlock  # 请替换成实际导入
        self.uib_end = UniversalInvertedBottleneckBlock(
            inp=(2+n)*self.c_mid,
            oup=c2,
            **uib_kwargs
        )

    def forward(self, x):
        y_split = self.cv1(x).chunk(2, dim=1)  # => (split1, split2)
        y = list(y_split)
        for block in self.m:
            y.append(block(y[-1]))  # 只处理最后那路
        out = torch.cat(y, dim=1)  # [B, 2*c_mid, H, W]
        # 用 UIB 做投影
        out = self.uib_end(out)
        return out
class C3k2_UIB_Replace(nn.Module):
    """
    改进思路2:
    直接用 UIB 替换掉C3k2中的重复Bottleneck/C3k。
    即: split -> [repeated UIB] -> concat -> 1x1conv (可选).
    """
    def __init__(self, c1, c2, n=1, c3k=False,e=0.5,
                 use_cv2=True,
                 uib_kwargs=None):
        """
        参数:
          c1, c2: 输入/输出通道数
          n: 重复UIB次数
          e: 中间通道= int(c2 * e)
          use_cv2: 是否保留末端1x1conv到c2；若False，则不做最终投影
          uib_kwargs: 传给UIB的dict
        """
        super().__init__()
        if uib_kwargs is None:
            uib_kwargs = {}
        # print(c2)
        self.c_mid = int(c2 * e)
        # print(self.c_mid)
        # 1) split
        self.cv1 = nn.Conv2d(c1, 2*self.c_mid, kernel_size=1, stride=1, bias=False)

        # 2) repeated UIB
        # from universal_inverted_bottleneck import UniversalInvertedBottleneckBlock
        self.m = nn.ModuleList()

        for _ in range(n):
            # UIB输入输出都是 self.c_mid
            block = UniversalInvertedBottleneckBlock(
                inp=self.c_mid,
                oup=self.c_mid,
                **uib_kwargs
            )
            self.m.append(block)

        # 3) concat后可选投影
        self.use_cv2 = use_cv2
        if self.use_cv2:
            self.cv2 = nn.Conv2d((2+n)*self.c_mid, c2, kernel_size=1, stride=1, bias=False)
        else:
            self.cv2 = nn.Identity()

    def forward(self, x):
        # split
        y_split = self.cv1(x).chunk(2, dim=1)  # => (split1, split2)
        y = list(y_split)

        # repeated UIB
        for block in self.m:
            y.append(block(y[-1]))

        out = torch.cat(y, dim=1)  # [B, 2*c_mid, H, W]
        out = self.cv2(out)        # [B, c2, H, W] or Identity
        return out

# 测试代码
if __name__ == "__main__":
    test_input = torch.randn(2, 64, 32, 32)
    lca_model = C3k2_UIB_Replace(64,64)
    with torch.no_grad():
        lca_out = lca_model(test_input)
    print(f"LCA Input shape: {test_input.shape}")
    print(f"LCA Output shape: {lca_out.shape}")