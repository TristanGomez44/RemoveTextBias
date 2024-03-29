import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import netBuilder
import sys
import torch
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,geom=False,geomConst=None,conv=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.lay1 and self.downsample layers downsample the input when stride != 1
        self.geom=geom

        if geom and not conv:
            self.lay1 = geomConst(inplanes,planes,batchNorm=False,boxPool=False,stride=stride)
            self.lay2 = geomConst(planes,planes,batchNorm=False,boxPool=False)
        elif conv and not geom:
            self.lay1 = conv3x3(inplanes, planes, stride)
            self.lay2 = conv3x3(planes, planes)
        elif geom and conv:
            self.geom1 = geomConst(inplanes,inplanes,batchNorm=False,boxPool=False)
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.lay1 = nn.Sequential(self.geom1,self.conv1)

            self.geom2 = geomConst(planes,planes,batchNorm=False,boxPool=False)
            self.conv2 = conv3x3(planes, planes)
            self.lay2 = nn.Sequential(self.geom2,self.conv2)

        else:
            raise ValueError("At least one of geom or conv must be true")

        self.actFunc = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.lay1(x)
        out = self.bn1(out)
        out = self.actFunc(out)

        out = self.lay2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.actFunc(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,geom=False,geomConst=None,conv=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if geom and not conv:
            self.lay2 = geomConst(width,width,batchNorm=False,boxPool=False,stride=stride)
        elif conv and not geom:
            self.lay2 = conv3x3(width, width, stride, groups)
        elif geom and conv:
            self.geom = geomConst(width,width,batchNorm=False,boxPool=False,stride=stride)
            self.conv = conv3x3(width, width, stride, groups)
            self.lay2 = nn.Sequential(self.geom,self.conv)
        else:
            raise ValueError("At least one of geom or conv must be true")

        self.lay1 = conv1x1(inplanes, width)
        self.lay3 = conv1x1(width, planes * self.expansion)
        self.actFunc = nn.ReLU(inplace=True)

        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.lay1(x)
        out = self.bn1(out)
        out = self.actFunc(out)

        out = self.lay2(out)
        out = self.bn2(out)
        out = self.actFunc(out)

        out = self.lay3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.actFunc(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None,geom=False,conv=True,inChan=3,\
                 strides=[2,2,2,2],firstConvKer=7,inPlanes=64,multiChan=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = inPlanes
        self.groups = groups
        self.base_width = width_per_group

        if firstConvKer%2 != 1:
            raise ValueError("The kernel size of the first convolution should be an odd number")
        self.geom = geom

        if multiChan:
            geomConst = netBuilder.GeomLayer_MC
        else:
            geomConst = netBuilder.GeomLayer

        if geom and not conv:
            self.lay1 = geomConst(inChan,self.inplanes,batchNorm=False,boxPool=False)
            self.actFunc = netBuilder.BoxPool(self.inplanes)
            self.maxpool = netBuilder.MaxPool2d_G((3,3))
        elif conv and not geom:
            self.lay1 = nn.Conv2d(inChan, self.inplanes, kernel_size=firstConvKer, stride=strides[0], padding=firstConvKer//2,bias=False)
            self.actFunc = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif conv and geom:
            self.geom1 = geomConst(inChan,inChan,batchNorm=False,boxPool=False)
            self.conv1 = nn.Conv2d(inChan, self.inplanes, kernel_size=firstConvKer, stride=strides[0], padding=firstConvKer//2,bias=False)
            self.lay1 = nn.Sequential(self.geom1,self.conv1)
            self.actFunc = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError("At least one of geom or conv must be true")

        self.bn1 = norm_layer(self.inplanes)

        self.layer1 = self._make_layer(block, inPlanes,  layers[0], stride=1, norm_layer=norm_layer,geom=geom,geomConst=geomConst,conv=conv)
        self.layer2 = self._make_layer(block, inPlanes*2, layers[1], stride=strides[1], norm_layer=norm_layer,geom=geom,geomConst=geomConst,conv=conv)
        self.layer3 = self._make_layer(block, inPlanes*4, layers[2], stride=strides[2], norm_layer=norm_layer,geom=geom,geomConst=geomConst,conv=conv)
        self.layer4 = self._make_layer(block, inPlanes*8, layers[3], stride=strides[3], norm_layer=norm_layer,geom=geom,geomConst=geomConst,conv=conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inPlanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,geom=False,geomConst=None,conv=True):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer,geom,geomConst,conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,geom=geom,geomConst=geomConst,conv=conv))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.lay1(x)
        x = self.bn1(x)

        featMaps = self.actFunc(x)
        if self.maxpool:
            featMaps = self.maxpool(featMaps)

        x = self.layer1(featMaps)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.geom:
            featMaps = x.clone()

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x,featMaps

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model
