import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.Auxiliary import Bottle_aux, DSC_aux


vgg_cfg = {
    11: [[64], ['M', 128,], ['M', 256, 256, ], ['M', 512, 512,], ['M', 512, 512]],
    13: [[64, 64,], ['M', 128, 128,],[ 'M', 256, 256,], ['M', 512, 512,], ['M', 512, 512]],
    16: [[64, 64,], ['M', 128, 128,], ['M', 256, 256, 256,], ['M', 512, 512, 512,], ['M', 512, 512, 512]],
    19: [[64, 64,], ['M', 128, 128,], ['M', 256, 256, 256, 256,],['M', 512, 512, 512, 512,], ['M', 512, 512, 512, 512]],
}


class VGG(nn.Module):
    def __init__(self, channels=3, num_classes=100, depth=16, bn=True, aux=None):
        super(VGG, self).__init__()

        print('| VGG-%d' %depth)
        cfg = vgg_cfg[depth]
        self.channels = channels
        self.cfg = cfg
        self.aux = aux
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.pre_layer = self.make_layers(self.channels, cfg[0], bn)
        self.layer_1 = self.make_layers(self.channels, cfg[1], batch_norm=bn)
        self.layer_2 = self.make_layers(self.channels, cfg[2], batch_norm=bn)
        self.layer_3 = self.make_layers(self.channels, cfg[3], batch_norm=bn)
        self.layer_4 = self.make_layers(self.channels, cfg[4], batch_norm=bn)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(128, 256, stride=2),
                Bottle_aux(256, 512, stride=2),
                Bottle_aux(512, 512, stride=2)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(256, 512, stride=2),
                Bottle_aux(512, 512, stride=2)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(512, 512, stride=2)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        elif self.aux =='G-Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(128, 256, stride=2, Groups=True),
                Bottle_aux(256, 512, stride=2, Groups=True),
                Bottle_aux(512, 512, stride=2, Groups=True)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(256, 512, stride=2, Groups=True),
                Bottle_aux(512, 512, stride=2, Groups=True)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(512, 512, stride=2, Groups=True)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        elif self.aux == 'DSC':
            self.aux_1_trans = nn.Sequential(
                DSC_aux(128, 256, stride=2),
                DSC_aux(256, 512, stride=2),
                DSC_aux(512, 512, stride=2)
            )
            self.linear_1 = nn.Linear(512, num_classes)

            self.aux_2_trans = nn.Sequential(
                DSC_aux(256, 512, stride=2),
                DSC_aux(512, 512, stride=2)
            )
            self.linear_2 = nn.Linear(512, num_classes)

            self.aux_3_trans = nn.Sequential(
                DSC_aux(512, 512, stride=2)
            )
            self.linear_3 = nn.Linear(512, num_classes)

        self._initialize_weights()

    def make_layers(self, in_channels,cfg, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.channels = cfg[-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        logits_list = []

        x = self.pre_layer(x)
        x = self.layer_1(x)
        if self.aux:
            out_1 = self.aux_1_trans(x)
            out_1 = self.relu(out_1)
            feature_list.append(out_1)
            out_1 = torch.flatten(self.avg_pool(out_1), 1)
            out_1 = self.linear_1(out_1)
            logits_list.append(out_1)

        x = self.layer_2(x)
        if self.aux:
            out_2 = self.aux_2_trans(x)
            out_2 = self.relu(out_2)
            feature_list.append(out_2)
            out_2 = torch.flatten(self.avg_pool(out_2), 1)
            out_2 = self.linear_2(out_2)
            logits_list.append(out_2)

        x = self.layer_3(x)
        if self.aux:
            out_3 = self.aux_3_trans(x)
            out_3 = self.relu(out_3)
            feature_list.append(out_3)
            out_3 = torch.flatten(self.avg_pool(out_3), 1)
            out_3 = self.linear_3(out_3)
            logits_list.append(out_3)

        x = self.layer_4(x)
        feature_list.append(x)
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        logits_list.append(out)

        if self.aux:
            return feature_list, logits_list
        else:
            return x, out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)  # 0.5
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def VGG16_bn(num_classes, aux):
    return VGG(num_classes=num_classes, depth=16, bn=True, aux=aux)


def VGG13_bn(num_classes, aux):
    return VGG(num_classes=num_classes, depth=13, bn=True, aux=aux)


def VGG19_bn(num_classes, aux):
    return VGG(num_classes=num_classes, depth=19, bn=True, aux=aux)




if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_bn(num_classes=100, aux='Bottle').cuda()
    # model = vgg19_bn(num_classes=100).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
