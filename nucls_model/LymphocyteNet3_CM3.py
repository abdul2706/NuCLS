import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, ReLU, Sequential, Linear, Dropout, Softmax
from nucls_model.ResNet import ResNet
from nucls_model.ResNetCBAM import ResNetCBAM

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

class LymphocyteNet3_CM3(Module):
    def __init__(self, debug, **kwargs):
        super(LymphocyteNet3_CM3, self).__init__()
        
        self.debug = debug
        self.module_name = 'LymphocyteNet3_CM3'
        
        self.backbone1 = ResNet(depth=50, debug=False)
        self.backbone2 = ResNetCBAM(depth=50, debug=False)

        self.block1 = self._make_layer(Bottleneck2, 256, 256, 2, stride=1) # 1 block inplace of 2
        self.block2 = self._make_layer(Bottleneck2, 512, 512, 2, stride=1)
        self.block3 = self._make_layer(Bottleneck2, 1024, 1024, 2, stride=1)
        self.block4 = self._make_layer(Bottleneck2, 2048, 2048, 2, stride=1, use_dropout=True)

        self.blocks = [self.block1, self.block2, self.block3, self.block4]

        self.out_channels = 2048

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, use_dropout=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.Dropout(p=0.35),
                nn.BatchNorm2d(planes),
            )

        layers = [block(inplanes, planes, stride, downsample, use_dropout=use_dropout, debug=False)]
        # inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes, use_dropout=use_dropout, debug=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}]', f'input to {self.module_name} | x.shape  =', x.shape, self.training)
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        # x3 = self.backbone3(x)

        if self.debug:
            print(f'[{self.module_name}]', 'output of backbone1 | ', end='')
            for level_out in x1:
                print(tuple(level_out.shape), end=' | ')
            print(f'\n[{self.module_name}]', 'output of backbone2 | ', end='')
            for level_out in x2:
                print(tuple(level_out.shape), end=' | ')
            print()

        features_merged = []
        # feature concatenation
        for i in range(4):
            if self.debug: print(f'[{self.module_name}]', i)
            features_add = x1[i] + x2[i]
            if self.debug: print(f'[{self.module_name}]', 'features_add.shape =', features_add.shape)
            features_merged.append(features_add)

        outs = []
        # feature reduction
        for i in range(4):
            ith_features = features_merged[i]
            if self.debug: print(f'[{self.module_name}]', i, ith_features.shape)
            ith_block = self.blocks[i]
            features_reduced = ith_block(ith_features)
            if self.debug: print(f'[{self.module_name}]', 'features_reduced.shape =', features_reduced.shape)
            outs.append(features_reduced)

        return outs

    # def init_weights(self, pretrained=None):
    #     print(f'[{self.module_name}]', 'pretrained -> ', pretrained)
    #     if pretrained == None: return
    #     pretrained = pretrained.split(';')
    #     print(f'[{self.module_name}]', 'pretrained -> ', pretrained)
    #     self.backbone1.init_weights(pretrained[0])
    #     self.backbone2.init_weights(pretrained[0])
