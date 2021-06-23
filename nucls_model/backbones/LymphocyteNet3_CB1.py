import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url
from torchvision import models
from . import ResNetCBAM

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ChannelAttention2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention2(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dropout=False, debug=False):
        super(Bottleneck2, self).__init__()

        self.module_name = 'Bottleneck2'
        self.debug = debug

        self.conv1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // 2)
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.conv3 = nn.Conv2d(planes // 2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.35) if use_dropout else None

        self.ca = ChannelAttention2(planes)
        self.sa = SpatialAttention2()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.debug: print(f'[{self.module_name}][1] x.shape -> {x.shape}')

        out = self.conv1(x)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn1(out)
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][2] out.shape -> {out.shape}')

        out = self.conv2(out)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn2(out)
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][3] out.shape -> {out.shape}')

        out = self.conv3(out)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn3(out)
        if self.debug: print(f'[{self.module_name}][4] out.shape -> {out.shape}')

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][8] out.shape -> {out.shape}')

        return out

class LymphocyteNet3_CB1(nn.Module):

    architectures = {
        18: [Bottleneck2, [64, 128, 256, 512]],
        34: [Bottleneck2, [64, 128, 256, 512]],
        50: [Bottleneck2, [256, 512, 1024, 2048]],
        101: [Bottleneck2, [256, 512, 1024, 2048]],
        152: [Bottleneck2, [256, 512, 1024, 2048]]
    }

    resnets = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, depth, use_dropout=False, pretrained=False, debug=False):
        super(LymphocyteNet3_CB1, self).__init__()

        self.module_name = 'LymphocyteNet3_CB1'
        self.depth = depth
        self.use_dropout = use_dropout
        self.pretrained = pretrained
        self.debug = debug
        block, planes = self.architectures[depth]
        self.out_channels = planes[-1]

        resnet = self.resnets[f'resnet{depth}'](pretrained)
        self.backbone1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.backbone2 = ResNetCBAM(depth=depth, use_dropout=use_dropout, pretrained=pretrained, debug=False)
        self.reducer1 = nn.Conv2d(512, 256, kernel_size=1)
        self.reducer2 = nn.Conv2d(512, 256, kernel_size=1)
        self.block4 = self._make_layer(block, planes[3], planes[3], 2, stride=1, use_dropout=use_dropout)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, use_dropout=False):
        downsample = None
        # if stride != 1 or inplanes != planes * block.expansion:
        #     if use_dropout:
        #         downsample = nn.Sequential(
        #             nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #             nn.Dropout(p=0.25),
        #             nn.BatchNorm2d(planes * block.expansion),
        #         )
        #     else:
        #         downsample = nn.Sequential(
        #             nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(planes * block.expansion),
        #         )

        layers = [block(inplanes, planes, stride, downsample, use_dropout=use_dropout, debug=False)]
        for i in range(1, blocks):
            layers.append(block(planes, planes, use_dropout=use_dropout, debug=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}]', f'input to {self.module_name} | x.shape  =', x.shape, self.training)
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        x1 = self.reducer1(x1)
        x2 = self.reducer2(x2)

        outs = torch.cat([x1, x2], dim=1)
        outs = self.block4(outs)
        # print(self.module_name, '[type(outs)][outs.shape]')
        # print(type(outs[0]), outs.shape)
        return outs
