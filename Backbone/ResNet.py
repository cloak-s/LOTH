import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from thop import clever_format, profile
from Backbone.Aux import Bottle_aux, SCAN_aux
from Backbone.Att import SCAN_Layer


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, aux=None, before_relu=False):
        super(_ResNet, self).__init__()
        if block == BasicBlock:
            depth = sum(num_blocks) * 2 + 2
        else:
            depth = sum(num_blocks) * 3 + 2
        print('| ResNet-%d-%s' % (depth, aux))

        self.aux = aux
        self.inplanes = 64
        self.before_relu = before_relu

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        if self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(64, 128, stride=2),
                Bottle_aux(128, 256, stride=2),
                Bottle_aux(256, 512, stride=2)
            )
            self.linear_1 = nn.Linear(512 * block.expansion, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(128, 256, stride=2),
                Bottle_aux(256, 512, stride=2)
            )
            self.linear_2 = nn.Linear(512 * block.expansion, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(256, 512, stride=2)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        elif self.aux == 'G-Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(64, 128, stride=2, Groups=True),
                Bottle_aux(128, 256, stride=2, Groups=True),
                Bottle_aux(256, 512, stride=2, Groups=True)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(128, 256, stride=2, Groups=True),
                Bottle_aux(256, 512, stride=2, Groups=True)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(256, 512, stride=2, Groups=True)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        elif self.aux == 'SCAN':
            self.aux_1_trans = nn.Sequential(
                SCAN_Layer(64),
                SCAN_aux(64, 512, 8)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                SCAN_Layer(128),
                SCAN_aux(128, 512, 4)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                SCAN_Layer(256),
                SCAN_aux(256, 512, 8)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        logits_list = []
        x = self.bn1(self.conv1(x))

        x = self.layer1(x)
        if self.aux:
            intem_1 = self.aux_1_trans(x)
            out_1 = self.relu(intem_1)
            if self.before_relu:
                feature_list.append(intem_1)
            else:
                feature_list.append(out_1)
            out_1 = torch.flatten(self.avg_pool(out_1), 1)
            out_1 = self.linear_1(out_1)
            logits_list.append(out_1)

        x = self.layer2(x)
        if self.aux:
            intem_2 = self.aux_2_trans(x)
            out_2 = self.relu(intem_2)
            if self.before_relu:
                feature_list.append(intem_2)
            else:
                feature_list.append(out_2)
            out_2 = torch.flatten(self.avg_pool(out_2), 1)
            out_2 = self.linear_2(out_2)
            logits_list.append(out_2)

        x = self.layer3(x)
        if self.aux:
            intem_3 = self.aux_3_trans(x)
            out_3 = self.relu(intem_3)
            if self.before_relu:
                feature_list.append(intem_3)
            else:
                feature_list.append(out_3)
            out_3 = torch.flatten(self.avg_pool(out_3), 1)
            out_3 = self.linear_3(out_3)
            logits_list.append(out_3)

        intem_4 = self.layer4(x)
        out_4 = self.relu(intem_4)
        if self.before_relu:
            feature_list.append(intem_4)
        else:
            feature_list.append(out_4)
        out = self.avg_pool(out_4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        logits_list.append(out)

        if self.aux:
            return feature_list, logits_list
        else:
            return out_4, out


def Res_14(num_classes, aux=None, before_relu=False):
    return _ResNet(BasicBlock, num_blocks=[1, 1, 1, 3], num_classes=num_classes, aux=aux, before_relu=before_relu)


def Res_18(num_classes, aux=None, before_relu=False):
    return _ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes, aux=aux, before_relu=before_relu)


def Res_34(num_classes, aux=None, before_relu=False):
    return _ResNet(BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=num_classes, aux=aux,before_relu=before_relu)


def Res_50(num_classes, aux=None, before_relu=False):
    return _ResNet(Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, aux=aux, before_relu=before_relu)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Res_50(num_classes=100, aux=None).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

