from thop import profile,clever_format
from torch import nn
import torch
import math
import torch.nn.functional as F
from models.Auxiliary import Bottle_aux
from models.Attention import DBSF_att, CS_att


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        # new_v += divisor
        res = new_v + divisor
        return res
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0, round_nearest=8, aux=None):
        super(MobileNetV2, self).__init__()
        self.aux = aux
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],

            [6, 32, 3, 2],

            [6, 64, 4, 2],
            [6, 96, 3, 1],

            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.first_layer = conv_3x3_bn(3, input_channel, 1)  # we change  stride to one
        index = 0
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        for t, c, n, s in interverted_residual_setting:
            if index <= 1:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layer1.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            elif 1 < index < 3:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layer2.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            elif 3 <= index < 5:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layer3.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            elif 5 <= index:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layer4.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            index = index + 1

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last_layer = conv_1x1_bn(input_channel, self.last_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.last_channel, num_classes)

        # if aux == 'DSC':
        #     self.aux_1_trans = nn.Sequential(
        #         DSC_aux(24, 256, 2),
        #         DSC_aux(256, 512, 2),
        #         DSC_aux(512, self.last_channel, 2)
        #     )
        #     self.linear_1 = nn.Linear(self.last_channel, num_classes)
        #
        #     self.aux_2_trans = nn.Sequential(
        #         DSC_aux(32, 512, 2),
        #         DSC_aux(512, self.last_channel, 2)
        #     )
        #     self.linear_2 = nn.Linear(self.last_channel, num_classes)
        #
        #     self.aux_3_trans = nn.Sequential(
        #         DSC_aux(96, self.last_channel, 2)
        #     )
        #     self.linear_3 = nn.Linear(self.last_channel, num_classes)

        if aux == 'G-Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 256, 2, Groups=True),
                Bottle_aux(256, 512, 2, Groups=True),
                Bottle_aux(512, self.last_channel, 2, Groups=True)
            )
            self.linear_1 = nn.Linear(self.last_channel, num_classes)

            self.aux_2_trans = nn.Sequential(

                Bottle_aux(32, 512, 2, Groups=True),
                Bottle_aux(512, self.last_channel, 2, Groups=True)
            )
            self.linear_2 = nn.Linear(self.last_channel, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(96, self.last_channel, 2, Groups=True)
            )
            self.linear_3 = nn.Linear(self.last_channel, num_classes)

        elif aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 256, 2, factor=2),
                Bottle_aux(256, 512, 2, factor=2),
                Bottle_aux(512, self.last_channel, 2, factor=2)
            )
            self.linear_1 = nn.Linear(self.last_channel, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(32, 512, 2, factor=2),
                Bottle_aux(512, self.last_channel, 2, factor=2)
            )
            self.linear_2 = nn.Linear(self.last_channel, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(96, self.last_channel, 2, factor=2)
            )
            self.linear_3 = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)

        x = self.layer1(x)
        if self.aux:
            out_1 = self.aux_1_trans(x)
            emb1 = out_1
            out_1 = torch.flatten(self.avg_pool(out_1), 1)
            out_1 = self.linear_1(out_1)

        x = self.layer2(x)
        if self.aux:
            out_2 = self.aux_2_trans(x)
            emb2 = out_2
            out_2 = torch.flatten(self.avg_pool(out_2), 1)
            out_2 = self.linear_2(out_2)

        x = self.layer3(x)
        if self.aux:
            out_3 = self.aux_3_trans(x)
            emb3 = out_3
            out_3 = torch.flatten(self.avg_pool(out_3), 1)
            out_3 = self.linear_3(out_3)

        x = self.layer4(x)
        x = torch.relu(x)
        x = self.last_layer(x)
        emb4 = x
        out = self.avg_pool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.aux:
            return [emb1, emb2, emb3, emb4], [out_1, out_2, out_3, out]
        else:
            return emb4, out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(num_classes=200, aux=None).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

# Ori  2.352M
# DSC  4.349M
# Ghost 6.405M
# Bottle 59.40
# G-Bottle 9.905M
