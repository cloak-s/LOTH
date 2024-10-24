"""shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Auxiliary import Bottle_aux, DSC_aux

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),  # 注意，这里没有relu的哦

                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU()
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU()
            )

        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )

    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)  # 这里作了一个通道shuffle

        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, ratio=1, num_classes=100, aux=None):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        print('| ShuffleNet-V2--{}'.format(aux))

        self.aux = aux
        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(out_channels[3], num_classes)

        if self.aux == 'G-Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 116, 2, Groups=True),
                Bottle_aux(116, 232, 2, Groups=True),
                Bottle_aux(232, 1024, 2,  Groups=True),
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(116, 232, 2, Groups=True),
                Bottle_aux(232, 1024, 2, Groups=True),
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(232, 1024, 2,  Groups=True)
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        elif self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 116, 2),
                Bottle_aux(116, 232, 2),
                Bottle_aux(232, 1024, 2),
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(116, 232, 2),
                Bottle_aux(232, 1024, 2),
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(232, 1024, 2)
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        elif self.aux == 'DSC':
            self.aux_1_trans = nn.Sequential(
                DSC_aux(24, 116, 2),
                DSC_aux(116, 232, 2),
                DSC_aux(232, 1024, 2),
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                DSC_aux(116, 232, 2),
                DSC_aux(232, 1024, 2),
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                DSC_aux(232, 1024, 2)
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        elif self.aux == 'DCM':
            self.aux_1_trans = nn.Sequential(
                self._make_stage(24, 116, 3),
                self._make_stage(116, 232, 7),
                self._make_stage(232, 1024, 3),
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                self._make_stage(116, 232, 7),
                self._make_stage(232, 1024, 3),
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                self._make_stage(232, 1024, 3),
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pre(x)
        if self.aux:
            out_1 = self.aux_1_trans(x)
            emb1 = out_1
            out_1 = torch.flatten(self.avg_pool(out_1), 1)
            out_1 = self.linear_1(out_1)

        x = self.stage2(x)
        if self.aux:
            out_2 = self.aux_2_trans(x)
            emb2 = out_2
            out_2 = torch.flatten(self.avg_pool(out_2), 1)
            out_2 = self.linear_2(out_2)

        x = self.stage3(x)
        if self.aux:
            out_3 = self.aux_3_trans(x)
            emb3 = out_3
            out_3 = torch.flatten(self.avg_pool(out_3), 1)
            out_3 = self.linear_3(out_3)

        x = self.stage4(x)
        x = self.conv5(x)
        emb4 = x
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.aux:
            return [emb1, emb2, emb3, emb4], [out_1, out_2, out_3, out]
        else:
            return emb4, out


    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)



if __name__ == "__main__":
    from thop import profile, clever_format
    x = torch.randn([1, 3, 32, 32])

    shuffleNet = ShuffleNetV2(ratio=1, num_classes=100, aux='DCM')
    print(shuffleNet)
    flops, params = profile(shuffleNet, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)






