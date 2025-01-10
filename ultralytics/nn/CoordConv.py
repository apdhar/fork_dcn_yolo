import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CoordConv(nn.Module):
    """CoordConv layer: Adds coordinate channels to the input tensor before applying a standard convolution."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize CoordConv with given arguments."""
        super().__init__()
        self.addcoords = AddCoords2D()
        self.conv = nn.Conv2d(c1 + 2, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply coordinate addition, convolution, batch normalization, and activation to input tensor."""
        x = self.addcoords(x)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply coordinate addition and convolution without batch normalization."""
        x = self.addcoords(x)
        return self.act(self.conv(x))


class AddCoords2D(nn.Module):
    """Adds x and y coordinate maps to the input tensor."""

    def forward(self, x):
        """Adds coordinate channels to input tensor."""
        b, _, h, w = x.size()
        device = x.device

        yy = torch.linspace(-1, 1, steps=h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, steps=w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, yy, xx], dim=1)




class C2f_CD(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CD_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CD_Bottleneck(nn.Module):
    """CoordConv-based bottleneck layer implementation."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=1.0):
        """Initialize CD_Bottleneck layer."""
        super().__init__()
        hidden_dim = int(c2 * e)
        self.cv1 = CoordConv(c1, hidden_dim, k[0], 1)
        self.cv2 = CoordConv(hidden_dim, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck operations."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
