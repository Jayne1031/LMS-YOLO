import torch
import torch.nn as nn


# -----------------------------
# 1) 低秩分解 1×1 卷积(加入BN + 激活)
# -----------------------------
class LowRankPointwiseConv2d(nn.Module):
    """
    逐点卷积 (Pointwise Conv) 的低秩分解:
    W ∈ R^{C_out x C_in} ~ U_r * V_r^T

    增加 BN + 激活函数:
      conv_reduce -> BN -> Act -> conv_expand -> BN -> Act
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        rank=2,
        bias=False,
        norm_layer=nn.BatchNorm2d,  # 可自定义
        act_layer=nn.ReLU           # 可自定义
    ):
        super(LowRankPointwiseConv2d, self).__init__()
        self.rank = rank

        # 第1步: Conv -> BN -> Act
        self.conv_reduce = nn.Conv2d(in_channels, rank, kernel_size=1, bias=bias)
        self.bn_reduce = norm_layer(rank)
        self.act_reduce = act_layer(inplace=True)

        # 第2步: Conv -> BN -> Act
        self.conv_expand = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)
        self.bn_expand = norm_layer(out_channels)
        self.act_expand = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.bn_reduce(x)
        x = self.act_reduce(x)

        x = self.conv_expand(x)
        x = self.bn_expand(x)
        x = self.act_expand(x)
        return x


# -----------------------------
# 2) 不对称空洞卷积(加入BN + 激活)
# -----------------------------
class AsymmetricDilatedConv2d(nn.Module):
    """
    不对称空洞卷积:
      - 分解 k x k 为 (1 x k) + (k x 1)
      - 每一步都可设空洞率 dilation
      - groups=in_channels 时即为深度卷积

    增加 BN + 激活函数:
      conv1 -> BN -> Act -> conv2 -> BN -> Act
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=2,
        bias=False,
        groups=1,
        norm_layer=nn.BatchNorm2d,  # 可自定义
        act_layer=nn.ReLU           # 可自定义
    ):
        super(AsymmetricDilatedConv2d, self).__init__()

        # 计算 padding；简单方式 = dilation * (kernel_size//2)
        padding = dilation * (kernel_size // 2)

        # 第一次卷积: (1 x k)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias
        )
        self.bn1 = norm_layer(out_channels)
        self.act1 = act_layer(inplace=True)

        # 第二次卷积: (k x 1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=groups,
            bias=bias
        )
        self.bn2 = norm_layer(out_channels)
        self.act2 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


# -----------------------------
# 3) 不对称空洞深度可分离 (DW + PW)
#    - 这里也可切换成新的带BN+激活的 AsymmetricDilatedConv2d
#    - 以及新的 LowRankPointwiseConv2d
# -----------------------------
class AsymmetricDilatedSeparableConv2d(nn.Module):
    """
    不对称空洞深度可分离卷积:
      - DW(AsymmetricDilatedConv2d, groups=in_channels)
      - PW(1×1 LowRankPointwiseConv2d)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=2,
        rank=2,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU
    ):
        super(AsymmetricDilatedSeparableConv2d, self).__init__()
        # 不对称空洞深度卷积
        self.asymmetric_dilated_dw = AsymmetricDilatedConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            groups=in_channels,  # 深度卷积
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        # 低秩分解 1x1 卷积
        self.pw = LowRankPointwiseConv2d(
            in_channels,
            out_channels,
            rank=rank,
            bias=bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

    def forward(self, x):
        x = self.asymmetric_dilated_dw(x)
        x = self.pw(x)
        return x


# -----------------------------
# 4) ECA 注意力
# -----------------------------
class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = t if (t % 2 == 1) else (t + 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        # 进行 1D 卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# -----------------------------
# 5) 动态核大小选择 (改进后版本)
# -----------------------------
class DynamicKernelSelection(nn.Module):
    """
    动态选择核大小的模块

    改进要点：
      1) 输入通道数 = in_channels（即 group_channels）
      2) 输出 shape = [N]，其中 N = B*groups
      3) 用 .multinomial 或 .argmax 进行选择
    """
    def __init__(self, in_channels, kernel_choices):
        super(DynamicKernelSelection, self).__init__()
        self.kernel_choices = kernel_choices
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(kernel_choices), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        x.shape = [N, in_channels, H, W]
          - 这里 N = B*groups
          - in_channels = group_channels
        """
        # 得到每个样本对所有 kernel_choices 的选择概率
        kernel_probs = self.predictor(x)
        # kernel_probs.shape = [N, len(kernel_choices), 1, 1]

        # 压缩掉最后两个维度 => [N, len(kernel_choices)]
        kernel_probs = kernel_probs.squeeze(-1).squeeze(-1)

        # 随机采样 => 每个样本(分组)选择1个 kernel index => [N]
        # 若想改为 “选择概率最大的 kernel”，可用 torch.argmax(kernel_probs, dim=1)
        selected_kernels = torch.multinomial(kernel_probs, num_samples=1).squeeze(1)
        return selected_kernels


# -----------------------------
# 6) 轻量化动态多尺度注意力模块（不对称空洞卷积版）
# -----------------------------
class LightweightDynamicLMSAWithAsymmetricDilatedConv(nn.Module):
    """
    在不对称空洞卷积 + 低秩分解 + 动态多尺度 + ECA 注意力的思路上做的模块
    可插入到 YOLOv8 颈部网络(C2f)中

    改进要点：
      1) 分组后，每个分组通道变为 group_channels
      2) DynamicKernelSelection 与分组后的特征匹配
      3) 对每个分组选出的 kernel 依次送入相应分支
      4) (可选) ECA 注意力在分组后 / 或在合并后
      5) (可选) 残差连接
    """
    def __init__(
        self,
        channels,
        factor=8,         # 分组数
        k_sizes=(3, 5, 7),
        rank=2,
        dilation=2,
        use_res=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU
    ):
        super().__init__()
        assert channels % factor == 0, "channels必须能被factor整除"
        self.groups = factor
        self.group_channels = channels // factor
        self.k_sizes = list(k_sizes)
        self.use_res = use_res

        # 动态核选择器: in_channels = group_channels
        self.kernel_selector = DynamicKernelSelection(
            in_channels=self.group_channels,
            kernel_choices=self.k_sizes
        )

        # 不同 kernel_size 建立不同分支
        self.dynamic_convs = nn.ModuleDict({
            str(k): AsymmetricDilatedSeparableConv2d(
                in_channels=self.group_channels,
                out_channels=self.group_channels,
                kernel_size=k,
                dilation=dilation,
                rank=rank,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for k in self.k_sizes
        })

        # ECA 注意力（作用在 group_channels 上）
        # 如果你想对合并后的全通道做 ECA，则可以把 ECA 移到后面
        self.attention = ECA(self.group_channels)

        # 如果想对合并后的 [B, c, H, W] 做 ECA，可以改成:
        # self.attention_full = ECA(channels)

    def forward(self, x):
        identity = x if self.use_res else None  # 残差
        b, c, h, w = x.shape

        # 1) 分组 => [B * groups, group_channels, H, W]
        group_x = x.view(b * self.groups, self.group_channels, h, w)

        # 2) 对每个分组（共 B*groups 个）预测 kernel 大小 => [B*groups]
        kernel_indices = self.kernel_selector(group_x)

        # 3) 逐分组送到相应分支
        out_list = []
        for i in range(b * self.groups):
            kernel_idx = kernel_indices[i].item()
            k_val = self.k_sizes[kernel_idx]
            branch = self.dynamic_convs[str(k_val)]
            # 当前分组特征 [1, group_channels, H, W]
            out_i = branch(group_x[i:i+1])
            out_list.append(out_i)

        # 4) 拼接 => [B*groups, group_channels, H, W]
        combined = torch.cat(out_list, dim=0)

        # 5) ECA 注意力（在分组后做）
        out = self.attention(combined)

        # 如果想对合并后的全通道 [B, c, H, W] 做 ECA, 可以先 reshape -> out = out.view(b, c, h, w)
        # 然后 out = self.attention_full(out)  # 若定义了 self.attention_full = ECA(channels)
        # 下方演示的是在这里先对分组结果做 ECA，再合并形状

        # 6) 还原形状 => [B, c, H, W]
        out = out.view(b, c, h, w)

        # 7) 残差连接 (可选)
        if self.use_res and identity is not None and identity.shape == out.shape:
            out = out + identity

        return out



# -----------------------------
# 7) 替换 YOLOv8 C2f 中的模块 (示例)
# -----------------------------
# class C2f_LMSA2(nn.Module):
#     """
#     一个示例: 替换 YOLOv8 中的 C2f，插入我们的
#     LightweightDynamicLMSAWithAsymmetricDilatedConv 模块
#     """
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n=2,
#         expansion=0.5,
#         factor=8,
#         k_sizes=(3, 5, 7),
#         rank=2,
#         dilation=2,
#         shortcut=True,
#         norm_layer=nn.BatchNorm2d,
#         act_layer=nn.ReLU
#     ):
#         """
#         参数说明(与YOLOv8 c2f类似):
#           in_channels  : 输入通道
#           out_channels : 输出通道
#           n            : 重复次数
#           expansion    : 通道扩展比例
#           factor       : 分组数
#           k_sizes      : 不同卷积核大小
#           rank         : 低秩分解秩
#           dilation     : 空洞率
#           shortcut     : 是否使用残差
#         """
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)
#         # 将输入拆分给两份(类似C2f的cv1, cv2)
#         self.cv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
#         self.cv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
#
#         # 堆叠 n 次我们的自定义模块
#         self.m = nn.Sequential(*[
#             LightweightDynamicLMSAWithAsymmetricDilatedConv(
#                 channels=hidden_channels,
#                 factor=factor,
#                 k_sizes=k_sizes,
#                 rank=rank,
#                 dilation=dilation,
#                 use_res=shortcut,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer
#             )
#             for _ in range(n)
#         ])
#         # 最后 1×1 卷积融合 (hidden_channels * (n + 1) => out_channels)
#         self.cv3 = nn.Conv2d(hidden_channels * (n + 1), out_channels, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         y1 = self.cv1(x)  # [B, hidden_channels, H, W]
#         y2 = self.cv2(x)  # [B, hidden_channels, H, W]
#         out_list = [y1]
#         out = y2
#         for module in self.m:
#             out = module(out)
#             out_list.append(out)
#
#         out = self.cv3(torch.cat(out_list, dim=1))  # concat后再1×1卷积
#         return out

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
class C2f_LMSA2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.emlsattention = LightweightDynamicLMSAWithAsymmetricDilatedConv(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # import pdb
        # pdb.set_trace()
        return self.emlsattention(out)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.emlsattention(out)
# -----------------------------
# 8)  测试示例
# -----------------------------
if __name__ == "__main__":
    x = torch.randn(8, 128, 32, 32)

    # 1) 测试单独的 LightweightDynamicLMSAWithAsymmetricDilatedConv
    lmsa_module = LightweightDynamicLMSAWithAsymmetricDilatedConv(
        channels=128,
        factor=8,
        k_sizes=[3, 5, 7],
        rank=2,
        dilation=2,
        use_res=True,  # 是否在内部加残差
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU
    )
    out_lmsa = lmsa_module(x)
    print("单模块输出形状:", out_lmsa.shape)

    # # 2) 测试插入到 C2f_LMSA
    # c2f_module = C2f_LMSA(
    #     in_channels=64, out_channels=64,
    #     n=2, expansion=0.5,
    #     factor=8, k_sizes=[3, 5, 7],
    #     rank=2, dilation=2,
    #     shortcut=True,
    #     norm_layer=nn.BatchNorm2d,
    #     act_layer=nn.ReLU
    # )
    # out_c2f = c2f_module(x)
    # print("C2f_LMSA 输出形状:", out_c2f.shape)
