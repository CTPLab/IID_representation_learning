import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class NoiseInjection(nn.Module):
    """ The injection of morph (noise of StyleGAN) representation
    to the backbone layers with different resolution 

    Args:
        lat_chn: the channel dimension of the morph (noise) represention
        drop: the dropout ratio or do not use dropout if False
    """

    def __init__(self, lat_chn, drop=False):
        super().__init__()

        self.lat_chn = lat_chn
        self.weight = nn.Parameter(torch.zeros(lat_chn))
        self.dropout = None if drop == 1 else nn.Dropout(p=drop, inplace=True)

    def forward(self, image, noise=None):
        if noise is None:
            return image
        else:
            noise = self.weight * noise
            if self.dropout is not None:
                noise = self.dropout(noise)
            return image + noise


class StyleInjection(nn.Module):
    """ The injection of stain (style of StyleGAN) representation
    for latent representation concatenation 

    Args:
        inp_chn: the channel dimension of the stain (style) represention
        drop: the dropout ratio or do not use dropout if False
    """

    def __init__(self, inp_chn, drop=False):
        super().__init__()

        self.conv = nn.Conv1d(inp_chn, 1, 3, 1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.leak = nn.LeakyReLU()
        self.dropout = None if drop == 1 else nn.Dropout(p=drop, inplace=True)

    def forward(self, style):
        style = self.conv(style)
        style = self.bn(style)
        style = self.leak(style)
        if self.dropout is not None:
            style = self.dropout(style)
        return style


class Model(nn.Module):
    """ The training model specification including 
    backbones, morph (noise) and stain (style) injection,
    etc.

    Args:
        args: critical parameters specified in args.py
        img_size: the input image size, this is used for 
            computing the stain (style) channel dim,
            by default is 256.
        style_dim: the spatial dimension of stain represention,
            by default is 512  
    """

    def __init__(self, args, img_size=256, style_dim=512):
        super().__init__()

        # Configure the four backbones investigated in the paper.
        # Since we want to inject morph to the layer with different resolution,
        # then the corresponding layers need to be decoupled as follows.
        backbone = args.backbone
        if backbone.startswith('densenet'):
            pretrained_backbone = getattr(
                torchvision.models, backbone)(pretrained=True)
            features_num = pretrained_backbone.features.norm5.num_features

            noise_dim = [64, 64, 128, 256, 512]

            layer0 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            layer1 = nn.Sequential(pretrained_backbone.features.norm0,
                                   pretrained_backbone.features.relu0,
                                   pretrained_backbone.features.pool0)
            layer2 = nn.Sequential(pretrained_backbone.features.denseblock1,
                                   pretrained_backbone.features.transition1)
            layer3 = nn.Sequential(pretrained_backbone.features.denseblock2,
                                   pretrained_backbone.features.transition2)
            layer4 = nn.Sequential(pretrained_backbone.features.denseblock3,
                                   pretrained_backbone.features.transition3)
            layer5 = nn.Sequential(pretrained_backbone.features.denseblock4,
                                   pretrained_backbone.features.norm5,
                                   nn.ReLU(inplace=True))

        elif backbone.startswith('resnet'):
            pretrained_backbone = getattr(
                torchvision.models, backbone)(pretrained=True)

            features_num = pretrained_backbone.fc.in_features

            noise_dim = [64, 64, 512, 1024, 2048]

            layer0 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            layer1 = nn.Sequential(pretrained_backbone.bn1,
                                   pretrained_backbone.relu,
                                   pretrained_backbone.maxpool)
            layer2 = nn.Sequential(pretrained_backbone.layer1,
                                   pretrained_backbone.layer2)
            layer3 = pretrained_backbone.layer3
            layer4 = pretrained_backbone.layer4
            layer5 = None

        elif backbone.startswith('mobilenet'):
            pretrained_backbone = getattr(
                torchvision.models, backbone)(pretrained=True)

            features_num = pretrained_backbone.features[-1][0].out_channels

            noise_dim = [32, 24, 32, 64, 160]

            first_conv = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            pretrained_backbone.features[0][0] = first_conv

            layer0 = pretrained_backbone.features[:1]
            layer1 = pretrained_backbone.features[1:3]
            layer2 = pretrained_backbone.features[3:5]
            layer3 = pretrained_backbone.features[5:8]
            layer4 = pretrained_backbone.features[8:15]
            layer5 = pretrained_backbone.features[15:]

        elif backbone.startswith('mnasnet'):
            pretrained_backbone = getattr(
                torchvision.models, backbone)(pretrained=True)

            features_num = pretrained_backbone.classifier[1].in_features

            noise_dim = [32, 24, 40, 80, 192]

            first_conv = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            pretrained_backbone = pretrained_backbone.layers
            pretrained_backbone[0] = first_conv

            layer0 = pretrained_backbone[:1]
            layer1 = pretrained_backbone[1:9]
            layer2 = pretrained_backbone[9:10]
            layer3 = pretrained_backbone[10:11]
            layer4 = pretrained_backbone[11:13]
            layer5 = pretrained_backbone[13:]
        else:
            raise ValueError('wrong backbone')

        self.features = [layer0,
                         layer1,
                         layer2,
                         layer3,
                         layer4]
        self.features = nn.ModuleList(self.features)
        if layer5 is not None:
            self.features.append(layer5)

        # Prepare the list of morph (noise) injection layers
        self.noises = nn.ModuleList()
        for _ in noise_dim:
            noise = NoiseInjection(1, args.noise_drop)
            self.noises.append(noise)

        # Prepare the stain (style) injection layer
        self.style = None
        if args.style:
            # style_chn is 14 because image_size is 256
            style_chn = int(math.log(img_size, 2)) * 2 - 2
            self.style = StyleInjection(style_chn, args.style_drop)
            features_num += style_dim

        self.neck = nn.Sequential(
            nn.BatchNorm1d(features_num),
            nn.Linear(features_num, args.embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(args.embedding_size),
            nn.Linear(args.embedding_size, args.embedding_size, bias=False),
            nn.BatchNorm1d(args.embedding_size),
        )

        # This is meant for Arcface loss
        self.arc_margin_product = ArcMarginProduct(
            args.embedding_size, args.classes)

        # This is meant for CrossEntropy loss
        self.crs_linear = nn.Linear(args.embedding_size, args.classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = args.bn_mom

    def embed(self, x):
        x_lat = x[1:]
        x = x[0]
        for f_id, _ in enumerate(self.features):
            x = self.features[f_id](x)
            if f_id < len(self.noises):
                x = self.noises[f_id](x, x_lat[f_id])

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.style is not None:
            x_sty = self.style(x_lat[-1])
            x = torch.cat([x, x_sty.view(x_sty.shape[0], -1)], dim=1)

        embedding = self.neck(x)
        return embedding


class ModelAndLoss(nn.Module):
    """ Initialize the model, ArcFace and CrossEntropy loss.

    Args:
        args: critical parameters specified in args.py
        img_size: the input image size, this is used for 
            computing the stain (style) channel dim,
            by default is 256.
        style_dim: the spatial dimension of stain represention,
            by default is 512  
        Both image_size and style_dim are relevant to 
            Restyle auto-encoder configurations.
    """

    def __init__(self, args, img_size, style_dim):
        super().__init__()

        self.args = args
        self.model = Model(args, img_size, style_dim)
        self.crit_arcface = ArcFaceLoss()
        self.crit_entropy = DenseCrossEntropy()
        self.coef = self.args.loss_coef

    def train_forward(self, x, y):
        embedding = self.model.embed(x)

        arc_logits = self.model.arc_margin_product(embedding)
        arc_loss = self.crit_arcface(arc_logits, y)

        crs_logits = self.model.crs_linear(embedding)
        crs_loss = self.crit_entropy(crs_logits, y)

        loss = crs_loss * (1 - self.coef) + arc_loss * self.coef
        acc = (crs_logits.max(1)[1] == y.max(1)[1]).float().mean().item()
        return loss, acc

    def eval_forward(self, x):
        embedding = self.model.embed(x)
        crs_logits = self.model.crs_linear(embedding)
        return crs_logits


class DenseCrossEntropy(nn.Module):
    """ The CrossEntropy loss that takes the one-hot
    vector of the gt label as the input, should be equivalent to the 
    standard CrossEntropy implementation. The one-hot vector
    is meant for the ArcFaceLoss and CutMix augmentation

    Args:
        x: the output of the model.
        target: the one-hot ground-truth label
    """

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    """ ArcFaceLoss, see the Fig.2 and Eq.3 in 
    https://arxiv.org/pdf/1801.07698.pdf

    Args:
        s: the scale factor on the output for computing
            CrossEntropy
        m: the margin penalty on the target (ground-truth label)
    """

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cos(phi_logits + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        # compute the phi for the gt label dimension
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    """ Process the latent vectors to output the cosine vector 
    for the follow-up ArcFaceLoss computation.

    Args:
        in_features: the column dimension of the weights,
            which is identical to the dim of latent vectors.
        out_features: the row dimension of the weights,
            which is identical to the number of classes.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
