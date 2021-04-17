import torch.nn as nn
import math


def conv3x1x1(in_planes, out_planes, stride=1):
    """ with padding """
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1),
            stride=(stride, 1, 1), padding=(1, 0, 0), bias=False)

def conv1x3x3(in_planes, out_planes, stride=1):
    """ with padding """
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
            stride=(1, stride, stride), padding=(0, 1, 1), bias=False)

class BasicConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicSTConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicSTConv3d, self).__init__()
        self.conv1 = conv1x3x3(in_planes, out_planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)

        self.conv2 = conv3x1x1(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.conv1_1 = conv1x3x3(in_planes, planes, stride=stride)
        self.bn1_1 = nn.BatchNorm3d(planes)
        self.conv1_2 = conv3x1x1(planes, planes, stride=stride)
        self.bn1_2 = nn.BatchNorm3d(planes)

        self.conv2_1 = conv1x3x3(planes, planes)
        self.bn2_1 = nn.BatchNorm3d(planes)
        self.conv2_2 = conv3x1x1(planes, planes)
        self.bn2_2 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residule = x

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)

        if self.downsample is not None:
            residule = self.downsample(residule)

        out = residule + x
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = BasicSTConv3d(planes, planes, stride=stride)
        #self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, 4 * planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * planes)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residule = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residule = self.downsample(residule)

        out += residule
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, block, layers, sample_duration=32, sample_size=224, num_classes=1000):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3,
                               bias=False)      #(b, 64, 32, 112, 112)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  #(b, 64, 16, 56, 56)
        self.layer1 = self._make_layer(block, 64, layers[0])             #(b, 256, 16, 56, 56)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  #(b, 512, 8, 28, 28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  #(b, 1024, 4, 14, 14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  #(b, 2048, 2, 7, 7)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model