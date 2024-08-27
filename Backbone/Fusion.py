import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from thop import profile, clever_format
from Backbone.Att import ECA_layer, SCAN_Layer, DECA_layer, JK_att, MCS_att


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
        

class Norm_fusion(nn.Module):  # FFL fusion
    def __init__(self, channel, num_class, layer, before_relu=False):
        super(Norm_fusion, self).__init__()

        self.relu = nn.ReLU()
        self.before_relu = before_relu
        # dw
        self.conv1 = nn.Conv2d(channel * layer, channel * layer, kernel_size=3, stride=1, padding=1,
                               groups=channel * layer, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * layer)

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
        if self.before_relu:
            embs = feats
        else:
            embs = self.relu(feats)
        out = self.relu(feats)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return embs, out


class DECA_fusion(nn.Module):
    def __init__(self, channel, num_class, layer, high=4):
        super(MECA_fusion, self).__init__()
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



class Logit_Fusion(nn.Module):
    def __init__(self, num_class):
        super(Logit_Fusion, self).__init__()
        # self.att = nn.Parameter(torch.ones(num_class, 1))
        self.att = nn.Parameter(torch.Tensor(num_class, 1))
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.att, gain=gain)

    def forward(self, Logits):
        Logits = torch.stack(Logits, dim=1)
        att_score = F.leaky_relu(torch.matmul(Logits, self.att))
        att_score = F.softmax(att_score, dim=1).permute(0, 2, 1)
        out = torch.matmul(att_score, Logits).squeeze(1)
        return out




if __name__ == '__main__':
    from thop import profile, clever_format

    inputs = []
    for i in range(4):
        inputs.append(torch.randn(10, 1280, 4, 4))

    # Fusion = Eca_fusion(channel=1280, num_class=100, layer=4)
    Fusion = Norm_fusion(num_class=100, channel=1280, layer=4, )
    print(Fusion)

    flops, params = profile(Fusion, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    print('ending')
