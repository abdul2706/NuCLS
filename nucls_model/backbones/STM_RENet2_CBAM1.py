import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # default kernel_size=7
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

class Attention(nn.Module):
    def __init__(self, planes, debug=False):
        super(Attention, self).__init__()
        self.module_name = 'Attention'
        self.debug = debug
        self.ca = ChannelAttention2(planes)
        self.sa = SpatialAttention2()

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}][1] x.shape -> {x.shape}')
        x = self.ca(x) * x
        if self.debug: print(f'[{self.module_name}][5] x.shape -> {x.shape}')
        x = self.sa(x) * x
        if self.debug: print(f'[{self.module_name}][6] x.shape -> {x.shape}')
        return x

class STM_RENet_BlockA(nn.Module):
    def __init__(self, in_channels, out_channels, debug=False):
        super(STM_RENet_BlockA, self).__init__()

        self.module_name = 'STM_RENet_BlockA'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        x = self.maxpool1(x)
        if self.debug: print(f'[{self.module_name}][maxpool1] x = {x.shape}')

        return x

class STM_RENet_BlockB(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, debug=False):
        super(STM_RENet_BlockB, self).__init__()
        
        self.module_name = 'STM_RENet_BlockB'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.attention = Attention(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=last_channels, kernel_size=1, padding=0, stride=1)

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

        x = self.attention(x)

        x = self.conv3(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        return x

class STM_RENet_BlockC(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, debug=False):
        super(STM_RENet_BlockC, self).__init__()
        
        self.module_name = 'STM_RENet_BlockC'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.attention = Attention(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=last_channels, kernel_size=1, padding=0, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        x = self.avgpool1(x)
        if self.debug: print(f'[{self.module_name}][avgpool1] x = {x.shape}')

        x = self.attention(x)

        x = self.conv2(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        return x

class STM_RENet_BlockD(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, debug=False):
        super(STM_RENet_BlockD, self).__init__()
        
        self.module_name = 'STM_RENet_BlockD'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, stride=1)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.attention = Attention(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=last_channels, kernel_size=1, padding=0, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        x = self.maxpool1(x)
        if self.debug: print(f'[{self.module_name}][maxpool1] x = {x.shape}')

        x = self.attention(x)

        x = self.conv2(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        return x

class STM_RENet2_CBAM1_Block(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, debug=False):
        super(STM_RENet2_CBAM1_Block, self).__init__()
        
        self.module_name = 'STM_RENet2_CBAM1_Block'
        self.debug = debug

        self.blockB1 = STM_RENet_BlockB(in_channels, out_channels, last_channels[0], debug=False)
        self.blockC1 = STM_RENet_BlockC(in_channels, out_channels, last_channels[1], debug=False)
        self.blockD1 = STM_RENet_BlockD(in_channels, out_channels, last_channels[2], debug=False)

    def forward(self, x):
        xB = self.blockB1(x)
        if self.debug: print(f'[{self.module_name}][blockB1] xB = {xB.shape}')
        xC = self.blockC1(x)
        if self.debug: print(f'[{self.module_name}][blockC1] xC = {xC.shape}')
        xD = self.blockD1(x)
        if self.debug: print(f'[{self.module_name}][blockD1] xD = {xD.shape}')

        out = torch.cat([xB, xC, xD], dim=1)
        # out = xB + xC + xD
        if self.debug: print(f'[{self.module_name}][out] out = {out.shape}')

        return out

class STM_RENet2_CBAM1(nn.Module):
    def __init__(self, debug=False):
        super(STM_RENet2_CBAM1, self).__init__()
        
        self.module_name = 'STM_RENet2_CBAM1'
        self.debug = debug
        self.out_channels = 512
        
        self.blockA = STM_RENet_BlockA(3, 32, debug=self.debug)
        self.layer1 = STM_RENet2_CBAM1_Block(32, 64, [22, 21, 21], debug=False)
        self.layer2 = STM_RENet2_CBAM1_Block(64, 128, [43, 43, 42], debug=False)
        self.layer3 = STM_RENet2_CBAM1_Block(128, 256, [86, 85, 85], debug=False)
        self.layer4 = STM_RENet2_CBAM1_Block(256, 512, [172, 170, 170], debug=False)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')
        x = self.blockA(x)
        if self.debug: print(f'[{self.module_name}][blockA] x = {x.shape}')

        layer1 = self.layer1(x)
        if self.debug: print(f'[{self.module_name}][layer1] layer1 = {layer1.shape}')
        
        layer2 = self.layer2(layer1)
        if self.debug: print(f'[{self.module_name}][layer2] layer2 = {layer2.shape}')

        layer3 = self.layer3(layer2)
        if self.debug: print(f'[{self.module_name}][layer3] layer3 = {layer3.shape}')

        layer4 = self.layer4(layer3)
        if self.debug: print(f'[{self.module_name}][layer4] layer4 = {layer4.shape}')

        return layer4

    # def init_weights(self, pretrained=None):
    #     print(f'{self.module_name} pretrained -> {pretrained}; training without pre-trained weights')
    #     if pretrained:
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
