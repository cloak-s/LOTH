"""
https://arxiv.org/abs/1611.05431
official code:
https://github.com/facebookresearch/ResNeXt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Backbone.VGG import Bottle_aRelu

from torch.autograd import Variable

"""
NOTICE:
    BasicBlock_B is not implemented
    BasicBlock_C is recommendation
    The full architecture consist of BasicBlock_A is not implemented.
"""


class ResBottleBlock(nn.Module):

    def __init__(self, in_planes, bottleneck_width=4, stride=1, expansion=1):
        super(ResBottleBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, bottleneck_width, 1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(bottleneck_width)
        self.conv1 = nn.Conv2d(bottleneck_width, bottleneck_width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_width)
        self.conv2 = nn.Conv2d(bottleneck_width, expansion * in_planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expansion, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_A(nn.Module):
    def __init__(self, in_planes, num_paths=32, bottleneck_width=4, expansion=1, stride=1):
        super(BasicBlock_A, self).__init__()
        self.num_paths = num_paths
        for i in range(num_paths):
            setattr(self, 'path' + str(i), self._make_path(in_planes, bottleneck_width, stride, expansion))

        # self.paths=self._make_path(in_planes,bottleneck_width,stride,expansion)
        self.conv0 = nn.Conv2d(in_planes * expansion, expansion * in_planes, 1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_planes * expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expansion, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.path0(x)
        for i in range(1, self.num_paths):
            if hasattr(self, 'path' + str(i)):
                out + getattr(self, 'path' + str(i))(x)
            # out+=self.paths(x)
            # getattr
        # out = torch.sum(out, dim=1)
        out = self.bn0(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def _make_path(self, in_planes, bottleneck_width, stride, expansion):
        layers = []
        layers.append(ResBottleBlock(
            in_planes, bottleneck_width, stride, expansion))
        return nn.Sequential(*layers)


class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0',
                 nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=10, aux=None):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        self.aux = aux
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        # self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)

        if self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aRelu(256, 512, stride=2, factor=1),
                Bottle_aRelu(512, 1024, stride=2, factor=1),
                Bottle_aRelu(1024, 2048, stride=2, factor=1)
            )
            self.linear_1 = nn.Linear(2048, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aRelu(512, 1024, stride=2, factor=1),
                Bottle_aRelu(1024, 2048, stride=2, factor=1)
            )
            self.linear_2 = nn.Linear(2048, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aRelu(1024, 2048, stride=2, factor=1)
            )
            self.linear_3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        feature_list = []
        logits_list = []

        x = F.relu(self.bn0(self.conv0(x)))
        # out = self.pool0(out)
        x = self.layer1(x)
        if self.aux:
            out_1 = self.aux_1_trans(x)
            out_1 = self.relu(out_1)
            feature_list.append(out_1)
            out_1 = torch.flatten(self.avg_pool(out_1), 1)
            out_1 = self.linear_1(out_1)
            logits_list.append(out_1)

        x = self.layer2(x)
        if self.aux:
            out_2 = self.aux_2_trans(x)
            out_2 = self.relu(out_2)
            feature_list.append(out_2)
            out_2 = torch.flatten(self.avg_pool(out_2), 1)
            out_2 = self.linear_2(out_2)
            logits_list.append(out_2)

        x = self.layer3(x)
        if self.aux:
            out_3 = self.aux_3_trans(x)
            out_3 = self.relu(out_3)
            feature_list.append(out_3)
            out_3 = torch.flatten(self.avg_pool(out_3), 1)
            out_3 = self.linear_3(out_3)
            logits_list.append(out_3)

        x = self.layer4(x)
        feature_list.append(x)
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        logits_list.append(out)

        if self.aux:
            return feature_list, logits_list
        else:
            return x, out

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)


def resnext26_2x64d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=2, bottleneck_width=64)


def resnext26_4x32d(num_class, aux):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=4, bottleneck_width=32, num_classes=num_class, aux=aux)


def resnext26_8x16d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=8, bottleneck_width=16)


def resnext26_16x8d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=16, bottleneck_width=8)


def resnext26_32x4d(num_class, aux):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4, num_classes=num_class, aux=aux)


def resnext26_64x2d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4)


def resnext50_2x64d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=2, bottleneck_width=64)


def resnext50_32x4d(num_class, aux):
    return ResNeXt(num_classes=num_class, num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, aux=aux)


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnext26_32x4d(aux='Bottle', num_class=100).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)