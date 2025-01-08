import torch
import torch.nn as nn

__all__ = ['BiFPN_Concat']

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        p = [(d * (x - 1) + 1 - x) // 2 for x in k] if isinstance(k, (list, tuple)) else (d * (k - 1) + 1 - k) // 2
    elif p is None:
        p = [(x // 2) for x in k] if isinstance(k, (list, tuple)) else k // 2
    return p

class Conv(nn.Module):
    """
    Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initializes a standard convolution layer with optional batch normalization and activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Applies a convolution followed by batch normalization and an activation function to the input tensor `x`.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Applies a fused convolution and activation function to the input tensor `x`.
        """
        return self.act(self.conv(x))

class BiFPN_Concat(nn.Module):
    """
    Concatenate a list of tensors along a dimension.
    """

    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Multi-layer addition or concatenation (1-3 layers).
        """
        if len(x) == 2:
            weight = self.w1_weight
            weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            weight = self.w2_weight
            weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x
