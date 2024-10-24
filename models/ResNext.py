import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbone.Aux import Bottle_aux, SCAN_aux
from Backbone.Att import SCAN_Layer

CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        C = CARDINALITY  #How many groups a feature map was splitted into

        D = int(DEPTH * out_channels / BASEWIDTH) #number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(),
            nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))


class ResNext(nn.Module):
    def __init__(self, block, num_blocks, num_classes, aux=None):
        super().__init__()

        self.in_channels = 64
        self.aux = aux
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 100)

        if self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(64, 128, stride=2),
                Bottle_aux(128, 256, stride=2),
                Bottle_aux(256, 512, stride=2)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(128, 256, stride=2),
                Bottle_aux(256, 512, stride=2)
            )
            self.linear_2 = nn.Linear(512, num_classes)

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

        elif self.aux == 'BYOT':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(64, 512, stride=8)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(128, 512, stride=4)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(256, 512, stride=2)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride

        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

# def resnext50():
#     """ return a resnext50(c32x4d) network
#     """
#     return ResNext(ResNextBottleNeckC, [3, 4, 6, 3])
#
# def resnext101():
#     """ return a resnext101(c32x4d) network
#     """
#     return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])
#
# def resnext152():
#     """ return a resnext101(c32x4d) network
#     """
#     return ResNext(ResNextBottleNeckC, [3, 4, 36, 3])


def ResNeXt_50(num_classes, aux=None):
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3], num_classes=num_classes, aux=aux)


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt_50(num_classes=100, aux=None).cuda()
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

