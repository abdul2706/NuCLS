import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url
from torchvision import models
from . import ResNetCBAM, STM_RENet2
# from .STM_RENet2 import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class STM_RENet_BlockB(nn.Module):
    def __init__(self, in_channels, out_channels, debug=False):
        super(STM_RENet_BlockB, self).__init__()
        
        self.module_name = 'STM_RENet_BlockB'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.avgpool(x)
        if self.debug: print(f'[{self.module_name}][avgpool1] x = {x.shape}')

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.maxpool(x)
        if self.debug: print(f'[{self.module_name}][maxpool1] x = {x.shape}')

        return x

class LymphocyteNet5_RE1(nn.Module):

    architectures = {
        18: [[64, 128, 256, 512]],
        34: [[64, 128, 256, 512]],
        50: [[256, 512, 1024, 2048]],
        101: [[256, 512, 1024, 2048]],
        152: [[256, 512, 1024, 2048]]
    }

    resnets = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, depth, use_dropout=False, pretrained=False, debug=False):
        super(LymphocyteNet5_RE1, self).__init__()

        self.module_name = 'LymphocyteNet5_RE1'
        self.depth = depth
        self.use_dropout = use_dropout
        self.pretrained = pretrained
        self.debug = debug
        planes = self.architectures[depth]
        self.out_channels = planes[-1]

        resnet = self.resnets[f'resnet{depth}'](pretrained)
        self.backbone1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.backbone2 = STM_RENet2(debug=False)
        # self.backbone2 = ResNetCBAM(depth=depth, use_dropout=use_dropout, pretrained=pretrained, debug=False)

        self.blockB1 = STM_RENet_BlockB(in_channels=512, out_channels=512, debug=True)
        self.blockB2 = STM_RENet_BlockB(in_channels=1024, out_channels=1024, debug=True)
        
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=2, stride=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(num_features=1024)

        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}]', f'input to {self.module_name} | x.shape  =', x.shape, self.training)
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        if self.debug: print(f'[{self.module_name}]', '[x1]', x1.shape)
        if self.debug: print(f'[{self.module_name}]', '[x2]', x2.shape)

        x1_b1 = self.blockB1(x1)
        x2_b1 = self.blockB1(x2)
        out = self.blockB2(torch.cat([x1_b1, x2_b1], dim=1))
        if self.debug: print(f'[{self.module_name}]', '[1][out]', out.shape)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}]', '[2][out]', out.shape)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}]', '[3][out]', out.shape)

        # print(self.module_name, '[type(outs)][outs.shape]')
        # print(type(outs[0]), outs.shape)
        return out
