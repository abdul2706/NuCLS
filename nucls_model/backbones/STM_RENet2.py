import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

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

        x = self.conv2(x)
        x = self.relu(x)
        if self.debug: print(f'[{self.module_name}] x = {x.shape}')

        return x

class STM_RENet2_Block(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, debug=False):
        super(STM_RENet2_Block, self).__init__()
        
        self.module_name = 'STM_RENet2_Block'
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

class STM_RENet2(nn.Module):
    def __init__(self, debug=False):
        super(STM_RENet2, self).__init__()
        
        self.module_name = 'STM_RENet2'
        self.debug = debug
        self.out_channels = 512
        
        self.blockA = STM_RENet_BlockA(3, 32, debug=self.debug)
        self.layer1 = STM_RENet2_Block(32, 64, [22, 21, 21], debug=False)
        self.layer2 = STM_RENet2_Block(64, 128, [43, 43, 42], debug=False)
        self.layer3 = STM_RENet2_Block(128, 256, [86, 85, 85], debug=False)
        self.layer4 = STM_RENet2_Block(256, 512, [172, 170, 170], debug=False)

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
    #     print(f'{self.module_name} pretrained -> {pretrained}')
    #     if pretrained:
    #         state_dict = load_state_dict_from_url(model_urls[f'resnet{self.depth}'], progress=True)
    #         self.load_state_dict(state_dict, strict=False)
