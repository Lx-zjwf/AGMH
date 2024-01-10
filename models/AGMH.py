from __future__ import absolute_import

import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
# from models.resnet import *
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import math
from collections import OrderedDict
from models.resnet_torch import resnet18, resnet50

# from ..utils.serialization import load_checkpoint, copy_state_dict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'SEMICON_backbone']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# stepwise interactive external attention
class SIExtAttn(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, dim, num_heads):
        super(SIExtAttn, self).__init__()

        self.num_heads = num_heads
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.k = 128

        self.qry_conv = nn.ModuleList([])
        for n in range(num_heads):
            self.qry_conv.append(nn.Conv1d(dim, self.k, 1, bias=False))

        self.alt_conv = nn.ModuleList([])
        for n in range(num_heads - 1):
            self.alt_conv.append(nn.Conv1d(2 * self.k, self.k, 1, bias=False))

        self.linear = nn.Conv1d(self.k, dim, 1, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim))
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        query_list = []
        for n in range(self.num_heads):
            query = self.qry_conv[n](x)  # b, k, n
            query_list.append(query)

        alt_query = query_list[0]
        for n in range(self.num_heads - 1):
            cat_query = torch.cat([alt_query, query_list[n + 1]], dim=1)
            alt_query = self.alt_conv[n](cat_query)

        attn = alt_query
        attn = F.softmax(attn, dim=-1)  # b, k, n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # b, k, n
        x = self.linear(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Refine(nn.Module):
    def __init__(self, feat_size, comp_size, top_k):
        super(Refine, self).__init__()
        self.top_k = top_k
        self.conv_list = nn.ModuleList([])
        self.attn_list = nn.ModuleList([])
        self.pool_list = nn.ModuleList([])
        for i in range(top_k + 1):
            self.conv_list.append(nn.Sequential(nn.Conv2d(feat_size, comp_size, 1),
                                                nn.BatchNorm2d(comp_size), nn.ReLU(inplace=True)))
            # MHExtAttn(dim=comp_size, num_heads=4) SpaAttn(comp_size) ExtAttn(c=comp_size)
            self.attn_list.append(CHExtAttn(dim=comp_size, num_heads=4))
            self.pool_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, device, is_train):
        feat_group = []
        attn_group = []
        vec_group = []
        for i in range(self.top_k + 1):
            conv_feat = self.conv_list[i](x)
            attn_feat = self.attn_list[i](conv_feat)
            pool_feat = self.pool_list[i](conv_feat)
            vec_feat = torch.flatten(pool_feat, 1)

            feat_group.append(conv_feat)
            attn_group.append(attn_feat)
            vec_group.append(vec_feat)

        return feat_group, attn_group, vec_group


class AGMH(nn.Module):
    def __init__(self, code_length=12, num_classes=200, att_size=3, feat_size=2048, device='cpu', pretrained=True):
        super(AGMH, self).__init__()
        self.top_k = 5
        self.device = device
        comp_size = int(feat_size / 4)  # (self.top_k + 1))
        rep_size = comp_size * (self.top_k + 1)
        self.backbone = resnet50(pretrained=pretrained)
        self.refine = Refine(feat_size, comp_size, self.top_k)

        self.hash_layer_active = nn.Sequential(nn.Tanh(), )
        self.code_length = code_length

        self.hash_map = nn.Parameter(torch.Tensor(code_length, rep_size))
        torch.nn.init.kaiming_uniform_(self.hash_map, a=math.sqrt(5))

    def forward(self, x, is_train=True):
        out = self.backbone(x)  # .detach()

        feat_group, attn_group, vec_group = self.refine(out, self.device, is_train)

        feat_vec = torch.cat(vec_group, dim=1)
        deep_hash = F.linear(feat_vec, self.hash_map)

        ret = self.hash_layer_active(deep_hash)

        return ret, out.detach(), feat_group, attn_group


def agmh(code_length, num_classes, att_size, feat_size, device, pretrained=False, **kwargs):
    model = AGMH(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs)
    return model
