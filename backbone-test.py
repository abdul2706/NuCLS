from nucls_model.backbones import *

import torch

TAG = '[backbone-test.py]'
x = torch.rand((1, 3, 224, 224))
print(TAG, '[x]', x.shape)
model = ResNet_RE1(depth=18, use_dropout=False, pretrained=False, conv_type='strided', debug=True)
print(TAG, '[model]')
print(model)
y = model(x)
print(TAG, '[y]', y.shape)
