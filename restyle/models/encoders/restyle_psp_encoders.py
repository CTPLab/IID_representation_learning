import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock, GradualNoiseBlock


class BackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """

    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(BackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], \
            'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class ResNetBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """

    def __init__(self, n_styles=18, opts=None, affine=True):
        super(ResNetBackboneEncoder, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=False,
                                  norm_layer=nn.InstanceNorm2d)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for _ in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

        self.noises = nn.ModuleList()
        self.noises_cnt = (n_styles - 1) // 2 - 1
        _inp_nc = 3
        _out_nc = 32
        _strd = 2
        for i in range(self.noises_cnt):
            noise = GradualNoiseBlock(_inp_nc, _out_nc, _strd, affine)
            self.noises.append(noise)
            _inp_nc = _out_nc
            _out_nc *= 2
        print(self.noises)

    def forward(self, x):
        sty = self.conv1(x)
        sty = self.bn1(sty)
        sty = self.relu(sty)
        sty = self.body(sty)
        styles = []
        for j in range(self.style_count):
            styles.append(self.styles[j](sty))
        styles = torch.stack(styles, dim=1)

        noise = x[:, :3].detach().clone()
        noises = list([None, None])
        for i in range(self.noises_cnt):
            noise, noise_out = self.noises[i](noise)
            noises = [noise_out, noise_out] + noises
        noises = [None] + noises
        return styles, noises
