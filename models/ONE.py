import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch
from thop import clever_format, profile
from models.ResNet import BasicBlock


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ONE_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, num_branches=3):
        super(ONE_ResNet, self).__init__()
        depth = sum(num_blocks) * 2 + 2
        print('| ONE_ResNet-%d' % depth)
        self.inplanes = 64
        self.num_branches = num_branches
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        fix_planes = self.inplanes
        for i in range(self.num_branches):
            setattr(self, 'layer4_'+str(i), self._make_layer(block, 512, num_blocks[3], stride=2))
            self.inplanes = fix_planes
            setattr(self, 'linear_'+str(i), nn.Linear(512 * block.expansion, num_classes))

        self.att_linear = nn.Linear(fix_planes, num_branches)  # 门控单元
        self.att_bn = nn.BatchNorm1d(num_branches)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_att = self.avg_pool(x)
        x_att = x_att.view(x_att.size(0), -1)
        x_att = self.att_linear(x_att)
        # x_att = self.att_bn(self.att_linear(x_att))  # 感觉这里的bn和relu都没有必要
        x_att = F.softmax(x_att, dim=1)

        logits = []
        for i in range(self.num_branches):
            out = getattr(self, 'layer4_'+str(i))(x)
            out = self.avg_pool(out)
            out = torch.flatten(out, 1)
            out = getattr(self, 'linear_'+str(i))(out)
            logits.append(out)

        out_t = x_att[:, 0].view(-1, 1) * logits[0]
        for i in range(1, len(logits)):
            out_t += x_att[:, i].view(-1, 1) * logits[i]
        return logits, out_t


def ONE_Res_18(num_classes, num_branches):
    return ONE_ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes, num_branches=num_branches)


def ONE_Res_34(num_classes, num_branches):
    return ONE_ResNet(BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=num_classes, num_branches=num_branches)


if __name__ == '__main__':
    import torch
    from thop import profile, clever_format

    x = torch.randn([10, 3, 32, 32])

    model = ONE_Res_34(100, 3)
    # print(model)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
