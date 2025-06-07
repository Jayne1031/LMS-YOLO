from timm.models import register_model

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import MODEL_SPECS


__all__ = ['mobilenetv4_small','mobilenetv4_medium', 'mobilenetv4_hybrid_large']

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


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1))
        self.block.add_module('conv_3x3',
                              conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

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


class LCA(nn.Module):
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

class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio,
                 add_LCA=False
                 ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        self.LCA = add_LCA
        if self.LCA:
            self.lca = LCA(oup)
        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # x = self.lca(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        if self.LCA:
            # print('add success!!!')
            x2 = x.clone()
            x = self.lca(x)
            x+=x2 # 残差连接
        else:
            x = x
        # print("_proj_conv", x.shape)
        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio','add_LCA']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, model):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"
        """
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]

        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # print(self.spec['layer4'])
        # layer5
        self.layer5 = build_blocks(self.spec['layer5'])
        self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        for f in self.features:
            # import pdb
            # pdb.set_trace()
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x

        return features



@register_model
def mobilenetv4_small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvSmall', **kwargs)
    return model


@register_model
def mobilenetv4_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvMedium', **kwargs)
    return model

@register_model
def mobilenetv4_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4ConvLarge', **kwargs)
    return model

@register_model
def mobilenetv4_hybrid_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4HybridMedium', **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = MobileNetV4('MobileNetV4HybridLarge', **kwargs)
    return model


# if __name__ == '__main__':
#     from torchinfo import summary
#     model = mobilenetv4_hybrid_large()
#     print("Check output shape ...")
#     summary(model, input_size=(1, 3, 224, 224))
    # x = torch.rand(1, 3, 224, 224)
    # y = model(x)
    # print(y.shape)
    # for i in y:
    #     print(i.shape)