import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp

# from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer, constant_init, kaiming_init)
# from mmcv.runner import load_checkpoint
# from mmdet.utils import get_root_logger

# from mmdet.models.builder import BACKBONES

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, debug=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckCBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dropout=False, debug=False):
        super(BottleneckCBAM, self).__init__()

        self.module_name = 'BottleneckCBAM'
        self.debug = debug

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2a = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(planes)
        self.conv2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) ### additional layer added
        self.bn2b = nn.BatchNorm2d(planes)
        self.conv2c = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  ###### additional layer added
        self.bn2c = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25) if use_dropout else None

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn1(out)
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')

        out = self.conv2a(out) # new added
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn2a(out)   # new added
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')
        out = self.conv2b(out)  # new added
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn2b(out)   # new added
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')
        out = self.conv2c(out)  # new added
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn2c(out)   # new added
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')
        out = self.relu(out)   # new added

        out = self.conv3(out)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn3(out)
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][1] out.shape -> {out.shape}')

        return out

class ResNetCBAM(nn.Module):

    architectures = {
        18: [BasicBlock, [2, 2, 2, 2]],
        34: [BasicBlock, [3, 4, 6, 3]],
        50: [BottleneckCBAM, [3, 4, 6, 3]],
        101: [BottleneckCBAM, [3, 4, 23, 3]],
        152: [BottleneckCBAM, [3, 8, 36, 3]]
    }
    
    def __init__(self, depth, debug=False):
        super(ResNetCBAM, self).__init__()

        self.module_name = 'ResNetCBAM'
        self.debug = debug
        self.inplanes = 64
        self.depth = depth
        block, layers = self.architectures[depth]
        # num_classes = 1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_dropout=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for layer in [self.layer1, self.layer2]:
            layer.training = False
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, use_dropout=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.Dropout(p=0.25),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, use_dropout=use_dropout, debug=False)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_dropout=use_dropout, debug=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}][1] x.shape -> {x.shape}')
        x = self.conv1(x)
        if self.debug: print(f'[{self.module_name}][2] x.shape -> {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.debug: print(f'[{self.module_name}][5] x.shape -> {x.shape}')

        level1 = self.layer1(x)
        if self.debug: print(f'[{self.module_name}][6] level1.shape -> {level1.shape}')
        level2 = self.layer2(level1)
        if self.debug: print(f'[{self.module_name}][7] level2.shape -> {level2.shape}')
        level3 = self.layer3(level2)
        if self.debug: print(f'[{self.module_name}][8] level3.shape -> {level3.shape}')
        level4 = self.layer4(level3)
        if self.debug: print(f'[{self.module_name}][9] level4.shape -> {level4.shape}')

        return (level1, level2, level3, level4)

    # def init_weights(self, pretrained=None):
    #     print(f'{self.module_name} pretrained -> {pretrained}')
    #     if pretrained:
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
