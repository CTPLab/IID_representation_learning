import numpy as np
from torch import nn
from torch.nn import Conv2d, Module

from models.stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)
        self.norm = nn.LayerNorm([out_c], elementwise_affine=False)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        x = self.norm(x)
        return x


class GradualNoiseBlock(Module):
    def __init__(self, in_c, out_c, stride, affine):
        super(GradualNoiseBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(out_c, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(1, affine=affine)
        self.downsample = nn.Conv2d(in_c, 1, kernel_size=3,
                                    stride=2, padding=1, bias=False)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        y = self.conv1(x) + identity
        y = self.norm1(y)
        return x, y
