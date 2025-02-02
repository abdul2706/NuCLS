import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
#     # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#     x_coord = torch.arange(kernel_size)
#     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#     y_grid = x_grid.t()
#     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

#     mean = (kernel_size - 1) / 2.
#     variance = sigma ** 2.

#     # Calculate the 2-dimensional gaussian kernel which is
#     # the product of two gaussian distributions for two different
#     # variables (in this case called x and y)
#     gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

#     # Make sure sum of values in gaussian kernel equals 1.
#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

#     # Reshape to 2d depthwise convolutional weight
#     gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
#     gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

#     if kernel_size == 3:
#         padding = 1
#     elif kernel_size == 5:
#         padding = 2
#     else:
#         padding = 0

#     gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False, padding=padding)
#     gaussian_filter.weight.data = gaussian_kernel
#     gaussian_filter.weight.requires_grad = False
    
#     return gaussian_filter

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dropout=False, conv_type='strided', debug=False):
        super(BasicBlock2, self).__init__()

        assert conv_type in ['strided', 'pooling'], f'Argument conv_type must be either `strided` or `pooling`, but got {conv_type} value'

        self.module_name = 'BasicBlock2'
        self.debug = debug
        self.planes = planes

        if stride == 1 or conv_type == 'strided':
            self.conv1 = conv3x3(inplanes, planes, stride)
        elif conv_type == 'pooling':
            self.conv1 = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=stride)
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout(p=0.25) if use_dropout else None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}][1] x.shape -> {x.shape}')
        residual = x

        out = self.conv1(x)
        if self.debug: print(f'[{self.module_name}][2] out.shape -> {out.shape}')
        # out = self.kernel1(out)
        out = self.dropout(out) if self.dropout is not None else out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.debug: print(f'[{self.module_name}][3] out.shape -> {out.shape}')
        # out = self.kernel2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.debug: print(f'[{self.module_name}][4] out.shape -> {out.shape}')

        return out

    # def get_new_kernels(self, kernel_size, std):
    #     self.kernel1 = get_gaussian_filter(kernel_size=kernel_size, sigma=std, channels=self.planes)
    #     self.kernel2 = get_gaussian_filter(kernel_size=kernel_size, sigma=std, channels=self.planes)

class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_dropout=False, conv_type='strided', debug=False):
        super(Bottleneck2, self).__init__()

        assert conv_type in ['strided', 'pooling'], f'Argument conv_type must be either `strided` or `pooling`, but got {conv_type} value'

        self.module_name = 'Bottleneck2'
        self.debug = debug
        self.planes = planes

        if stride == 1 or conv_type == 'strided':
            self.conv1 = conv3x3(inplanes, planes, stride)
        elif conv_type == 'pooling':
            self.conv1 = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=stride)
            )
        self.bn1 = nn.BatchNorm2d(planes)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  ### here orginal is stride=stride
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25) if use_dropout else None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.kernel1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out) if self.dropout is not None else out

        # here added pooling
        # out = self.pool(out) if self.pool is not None else out

        out = self.conv2(out)
        # out = self.kernel2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out) if self.dropout is not None else out

        out = self.conv3(out)
        # out = self.kernel3(out)
        out = self.bn3(out)
        out = self.dropout(out) if self.dropout is not None else out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout(out) if self.dropout is not None else out

        return out

    # def get_new_kernels(self, kernel_size, std):
    #     self.kernel1 = get_gaussian_filter(kernel_size=kernel_size, sigma=std, channels=self.planes)
    #     self.kernel2 = get_gaussian_filter(kernel_size=kernel_size, sigma=std, channels=self.planes)
    #     self.kernel3 = get_gaussian_filter(kernel_size=kernel_size, sigma=std, channels=self.planes*4)

class ResNet2(nn.Module):

    architectures = {
        18: [BasicBlock2, [2, 2, 2, 2], [64, 128, 256, 512]],
        34: [BasicBlock2, [3, 4, 6, 3], [64, 128, 256, 512]],
        50: [Bottleneck2, [3, 4, 6, 3], [256, 512, 1024, 2048]],
        101: [Bottleneck2, [3, 4, 23, 3], [256, 512, 1024, 2048]],
        152: [Bottleneck2, [3, 8, 36, 3], [256, 512, 1024, 2048]],
    }
    
    def __init__(self, depth=18, use_dropout=False, pretrained=False, conv_type='strided', debug=False):
        super(ResNet2, self).__init__()

        assert conv_type in ['strided', 'pooling'], f'Argument conv_type must be either `strided` or `pooling`, but got {conv_type} value'

        self.module_name = 'ResNet2'
        self.depth = depth
        self.pretrained = pretrained
        self.conv_type = conv_type
        self.debug = debug
        # self.kernel_size = 3
        # self.std = 1
        # self.epoch = 1
        # self.factor = 0.925
        block, blocks, channels = self.architectures[depth]
        self.inplanes = channels[0]
        self.out_channels = channels[3]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], blocks[0], stride=1, use_dropout=use_dropout, conv_type=conv_type)
        self.layer2 = self._make_layer(block, channels[1], blocks[1], stride=2, use_dropout=use_dropout, conv_type=conv_type)
        self.layer3 = self._make_layer(block, channels[2], blocks[2], stride=2, use_dropout=use_dropout, conv_type=conv_type)
        self.layer4 = self._make_layer(block, channels[3], blocks[3], stride=2, use_dropout=use_dropout, conv_type=conv_type)

        # self.layer1 = self._make_layer(block,  64, blocks[0], stride=1, use_dropout=use_dropout)
        # self.layer2 = self._make_layer(block, 128, blocks[1], stride=2, use_dropout=use_dropout)
        # self.layer3 = self._make_layer(block, 256, blocks[2], stride=2, use_dropout=use_dropout)
        # self.layer4 = self._make_layer(block, 512, blocks[3], stride=2, use_dropout=use_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for layer in [self.layer1, self.layer2]:
        #     layer.training = False
        #     layer.eval()
        #     for param in layer.parameters():
        #         param.requires_grad = False

        self.init_weights(self.pretrained)
        # self.get_new_kernels(0)

    def _make_layer(self, block, planes, blocks, stride=1, use_dropout=False, conv_type='strided'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if conv_type == 'strided':
                conv_layer = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            else:
                conv_layer = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=stride)
                )

            if use_dropout:
                downsample = nn.Sequential(
                    conv_layer,
                    nn.Dropout(p=0.25),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv_layer,
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, use_dropout=use_dropout, conv_type=conv_type, debug=False)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_dropout=use_dropout, conv_type=conv_type, debug=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug: print(f'[{self.module_name}][1] x.shape -> {x.shape}')
        x = self.conv1(x)
        # x = self.kernel1(x)
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

        return level4

    def init_weights(self, pretrained=None):
        print(f'{self.module_name} pretrained -> {pretrained}')
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[f'resnet{self.depth}'], progress=True)
            self.load_state_dict(state_dict, strict=False)
    
    # def get_new_kernels(self, epoch_count):
    #     if epoch_count % self.epoch == 0 and epoch_count is not 0:
    #         self.std *= self.factor
    #     self.kernel1 = get_gaussian_filter(kernel_size=self.kernel_size, sigma=self.std, channels=64)

    #     for child in self.layer1.children():
    #         child.get_new_kernels(self.kernel_size, self.std)

    #     for child in self.layer2.children():
    #         child.get_new_kernels(self.kernel_size, self.std)

    #     for child in self.layer3.children():
    #         child.get_new_kernels(self.kernel_size, self.std)

    #     for child in self.layer4.children():
    #         child.get_new_kernels(self.kernel_size, self.std)

if __name__ == '__main__':
    x = torch.rand((2, 3, 224, 224))
    print('[x]\n', x.shape)
    model = ResNet2(depth=18, use_dropout=False, pretrained=False, conv_type='strided', debug=False)
    print('[model]\n', model)
    y = model(x)
    print('[y]\n', y.shape)
