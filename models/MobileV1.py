from thop import profile, clever_format
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Auxiliary import Bottle_aux, DSC_aux
from models.Attention import SD_Layer, DBSF_att, CS_att, MCS_att
from Backbone.BYOT import branchBottleNeck


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(),
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100, aux=None):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        self.aux = aux
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        print('| Mobile-{}'.format(aux))

        self.conv1 = conv_bn(3, 32, 1)
        self.layer1 = nn.Sequential(
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1))

        self.layer2 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1))

        self.layer3 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1))

        self.layer4 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )
        self.fc = nn.Linear(1024, self.num_classes)

        if self.aux == 'ECSD':
            self.aux_1_trans = nn.Sequential(
                SD_Layer(128),
                DSC_aux(128, 256, 2),
                DSC_aux(256, 512, 2),
                DSC_aux(512, 1024, 2)
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                SD_Layer(256),
                DSC_aux(256, 512, 2),
                DSC_aux(512, 1024, 2)
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                SD_Layer(512),
                DSC_aux(512, 1024, 2)
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        elif self.aux == 'G-Bottle':
            self.aux_1_trans = nn.Sequential(
                # MCS_att(128),
                Bottle_aux(128, 256, 2, Groups=True),
                Bottle_aux(256, 512, 2, Groups=True),
                Bottle_aux(512, 1024, 2, Groups=True)
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                # MCS_att(256),
                Bottle_aux(256, 512, 2, Groups=True),
                Bottle_aux(512, 1024, 2, Groups=True)
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                # MCS_att(512),
                Bottle_aux(512, 1024, 2, Groups=True)
            )
            self.linear_3 = nn.Linear(1024, num_classes)


        elif self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(128, 256, 2),
                Bottle_aux(256, 512, 2),
                Bottle_aux(512, 1024, 2)
            )
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(256, 512, 2),
                Bottle_aux(512, 1024, 2)
            )
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(512, 1024, 2)
            )
            self.linear_3 = nn.Linear(1024, num_classes)

        elif self.aux == 'BYOT':
            self.aux_1_trans = branchBottleNeck(128, 1024, kernel_size=8)
            self.linear_1 = nn.Linear(1024, num_classes)

            self.aux_2_trans = branchBottleNeck(256, 1024, kernel_size=4)
            self.linear_2 = nn.Linear(1024, num_classes)

            self.aux_3_trans = branchBottleNeck(512, 1024, kernel_size=2)
            self.linear_3 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.conv1(x)
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
        emb4 = x
        out = self.avg_pool(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if self.aux:
            return [emb1, emb2, emb3, emb4], [out_1, out_2, out_3, out]
        else:
            return emb4, out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1(num_classes=200, aux=None).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

# 参数量 3.31M

# Bottle   42.87M
# G-Bottle 9.29M
# Ghost    6.52M
# DSC      5.517M
