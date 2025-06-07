import torch
import torch.nn as nn


class MGAM(nn.Module):
    def __init__(self, in_channels, rate=4, k_size=7):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)
        self.k_size = k_size

        # Channel attention layers
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)

        # Spatial attention layers
        self.conv0h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2),
                                groups=in_channels)
        self.conv0v = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0),
                                groups=in_channels)

        if k_size == 7:
            self.conv_spatial_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2),
                                            groups=in_channels, dilation=2)
            self.conv_spatial_v = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),
                                            groups=in_channels, dilation=2)
        elif k_size == 11:
            self.conv_spatial_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4),
                                            groups=in_channels, dilation=2)
            self.conv_spatial_v = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0),
                                            groups=in_channels, dilation=2)

        # Adapting the features
        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, 1)
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, 1)

        self.norm1 = nn.BatchNorm2d(inchannel_rate)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # Channel attention mechanism
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        # Spatial attention mechanism with multi-scale convolutions
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        x_spatial_att = self.relu(self.norm1(attn))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))

        # Final output with channel and spatial attention combined
        out = u * x_spatial_att

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



class C2f_MGAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.mgam = MGAM(c2)

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