import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from thop import profile, clever_format
from models.Attention import MCS_att, MS_CAM, DECA_layer, MECA_layer, ECA_layer
from models.Auxiliary import Bottle_aux
from models.ShuffleNetV2 import channel_shuffle


class Norm_fusion(nn.Module):  #
    def __init__(self, channel, num_class, layer):
        super(Norm_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channel * layer, channel * layer, kernel_size=3, stride=1, padding=1,
                               groups=channel * layer, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * layer)

        # self.att = DECA_layer(channel * layer, high=high)

        # point-wise conv
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.relu(self.bn1(self.conv1(feats)))
        feats = self.bn2(self.conv2(feats))
        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out


class KFD_Fusion(nn.Module):
    def __init__(self, channel, layer, num_class):
        super(KFD_Fusion, self).__init__()
        hidden = channel // 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.FC1 = nn.Linear(channel, hidden)
        self.BN = nn.BatchNorm1d(hidden)
        self.FC2 = nn.Linear(hidden, layer)
        self.CONV = nn.Conv2d(channel, channel, bias=False, kernel_size=1)
        self.BN2 = nn.BatchNorm2d(channel)
        self.classifier = nn.Linear(channel, num_class)

    def forward(self, feats):
        ref = sum(feats)
        ref = self.avg_pool(ref)
        ref = ref.view(ref.size(0), -1)
        att = self.FC2(F.relu(self.BN(self.FC1(ref))))
        att = F.softmax(att, dim=-1)
        fuse_feat = feats[0] * att[:, 0].view(-1, 1, 1, 1)
        for i in range(1, len(feats)):
            fuse_feat += feats[i] * att[:, i].view(-1, 1, 1, 1)
        fuse_feat = F.relu(self.BN2(self.CONV(fuse_feat)))
        # fuse_feat = self.avg_pool(fuse_feat)
        out = self.avg_pool(fuse_feat)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return fuse_feat, out


class DualNet_Fusion(nn.Module):
    def __init__(self, channel, num_class, layer):
        super(DualNet_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channel, num_class)
        self.conv = nn.Conv2d(channel, channel, bias=False, kernel_size=1)
        self.BN = nn.BatchNorm2d(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = sum(feats)
        feats = self.BN(self.conv(feats))
        feats = F.relu(feats)
        out = self.avg_pool(feats)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return feats, out


class EML_fusion(nn.Module):
    def __init__(self, channel, num_class, layer):
        super(EML_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(channel * layer, channel, kernel_size=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = [self.avg_pool(feat) for feat in feats]
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.bn(self.conv(feats))
        out = F.relu(feats)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return feats, out


class MECA_fusion(nn.Module):
    def __init__(self, channel, num_class, layer, high=4):
        super(MECA_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.att = MECA_layer(channel*layer, high=high)
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.att(feats)
        feats = self.bn2(self.conv2(feats))

        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out



class DECA_fusion(nn.Module):
    def __init__(self, channel, num_class, layer, high=4):
        super(DECA_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.att = DECA_layer(channel*layer, high=high)  # note that
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.att(feats)
        feats = self.bn2(self.conv2(feats))

        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out


class ECA_fusion(nn.Module):
    def __init__(self, channel, num_class, layer, hight=4):
        super(ECA_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.att = ECA_layer(channel)

        # point-wise conv
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.att(feats)
        feats = self.bn2(self.conv2(feats))

        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out


class LECA_fusion(nn.Module):
    def __init__(self, channel, num_class, layer, hight=4):
        super(LECA_fusion, self).__init__()
        self.relu = nn.ReLU()
        self.att = Local_ECA(channel, high=hight)

        # point-wise conv
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.att(feats)
        feats = self.bn2(self.conv2(feats))

        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out


class Simple_fusion(nn.Module):  # 仅 concatenate
    def __init__(self, channel, num_class, layer):
        super(Simple_fusion, self).__init__()
        self.relu = nn.ReLU()
        # point-wise conv
        self.conv2 = nn.Conv2d(channel * layer, channel, kernel_size=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channel, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        feats = torch.cat(feats, dim=1)  # concatenate
        feats = self.bn2(self.conv2(feats))
        out = self.relu(feats)
        embs = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out



if __name__ == '__main__':
    from thop import profile, clever_format

    inputs = []
    for i in range(4):
        inputs.append(torch.randn(10, 512, 4, 4))

    # Fusion = JK_Fusion(channel=1280, num_class=100, layer=4)
    Fusion = MECA_fusion(channel=512, num_class=100, layer=4)
    print(Fusion)
    # Fusion = Norm_fusion(channel=1280, num_class=100, layer=4)
    # Fusion = Msa_Fusion(channel=1280, num_class=100, layer=4, groups=64)

    flops, params = profile(Fusion, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    print('ending')
