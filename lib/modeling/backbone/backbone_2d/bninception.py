import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .utils import load_state_dict_from_url


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

### copy from CoogleLeNet
class BNInception(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=False, init_weights=True):
        super(BNInception, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 64, 64, 64, 96, 96, 32)
        self.inception3b = Inception(256, 64, 64, 96, 64, 96, 96, 64)
        self.inception3c = InceptionB(320, 128, 160, 64, 96, 96)
        #self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(576, 224, 64, 96, 96, 128, 128, 128)
        self.inception4b = Inception(576, 192, 96, 128, 96, 128, 128, 128)
        self.inception4c = Inception(576, 160, 128, 160, 128, 160, 160, 128)
        self.inception4d = Inception(608, 96, 128, 192, 160, 192, 192, 128)
        self.inception4e = InceptionB(608, 128, 192, 192, 256, 256)
        #self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(1056, 352, 192, 320, 160, 224, 224, 128)
        self.inception5b = Inception(1024, 352, 192, 320, 192, 224, 224, 128, last=True)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=True)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # N x 3 x 224 x 224
        x = self.conv1(x)
        print('shape after conv1:', x.shape)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        print('shape before inception branch:', x.shape)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        print('shape after inception3a:', x.shape)
        # N x 224 x 28 x 28
        x = self.inception3b(x)
        print('shape after inception3b:', x.shape)
        # N x 320 x 28 x 28
        x = self.inception3c(x)
        print('shape after inception3c:', x.shape)
        # N x 480 x 28 x 28
        #x = self.maxpool3(x)
        # N x 576 x 14 x 14
        x = self.inception4a(x)
        print('shape after inception4a (reduction):', x.shape)
        # N x 512 x 14 x 14
        # if self.training and self.aux_logits:
        #     aux1 = self.aux1(x)

        # N x 576 x 14 x 14
        x = self.inception4b(x)
        print('shape after inception4b:', x.shape)
        # N x 576 x 14 x 14
        x = self.inception4c(x)
        print('shape after inception4c:', x.shape)
        # N x 608 x 14 x 14
        x = self.inception4d(x)
        print('shape after inception4d:', x.shape)
        # N x 528 x 14 x 14
        # if self.training and self.aux_logits:
        #     aux2 = self.aux2(x)

        # N x 608 x 14 x 14
        x = self.inception4e(x)
        print('shape after inception4e (reduction):', x.shape)
        # N x 832 x 14 x 14
        #x = self.maxpool4(x)
        # N x 1056 x 7 x 7
        x = self.inception5a(x)
        print('shape after inception5a:', x.shape)
        # N x 1024 x 7 x 7
        x = self.inception5b(x)
        print('shape after inception5b:', x.shape)
        # N x 1024 x 7 x 7

        x = self.global_pool(x)
        print('shape after global pooling:', x.shape)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # if self.training and self.aux_logits:
        #     return _GoogLeNetOuputs(x, aux2, aux1)
        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5_mid, ch5x5_bot, pool_proj, last=False):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5_mid, kernel_size=3, padding=1),
            BasicConv2d(ch5x5_mid, ch5x5_bot, kernel_size=3, padding=1)
        )

        if not last:
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                BasicConv2d(in_channels, pool_proj, kernel_size=1)
            )
        else:
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                BasicConv2d(in_channels, pool_proj, kernel_size=1)
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):     # reduction block

    def __init__(self, in_channels, ch3x3red, ch3x3, ch5x5red, ch5x5_mid, ch5x5_bot):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5_mid, kernel_size=3, padding=1),
            BasicConv2d(ch5x5_mid, ch5x5_bot, kernel_size=3, padding=1, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.fc2(x)
        # N x 1024

        return x


def bninception(pretrained=True, checkpoint=None, **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = BNInception(**kwargs)
        #state_dict = load_state_dict_from_url(model_urls['googlenet'])
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return BNInception(**kwargs)
