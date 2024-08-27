import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SCAN_Layer(nn.Module):  # from 2019 SCAN
    def __init__(self, inp, stride=2):
        super(SCAN_Layer, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=stride, in_channels=inp,
                      out_channels=inp),
            nn.BatchNorm2d(inp),
            nn.ReLU(),
        )

        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=inp,
                               out_channels=inp),
            nn.BatchNorm2d(inp),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.down_sampling(x)
        x = self.up_sampling(x)
        return x * identity


class SD_Layer(nn.Module):  # from 2022 SD
    def __init__(self, channel, factor=2):
        super(SD_Layer, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, groups=channel, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(),

            nn.Conv2d(channel, channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.scale_factor = factor
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.down_sampling(x)
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.sigmoid(x)
        return x * identity


class SA_layer(nn.Module):    # shuffle attention
    def __init__(self, channel, groups=64):
        super(SA_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        # groups
        x = x.reshape(b * self.groups, -1, h, w)

        # split
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out



class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA_layer(nn.Module):
    def __init__(self, channel, gama=2, b=1):
        super(ECA_layer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gama))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class MECA_layer(nn.Module):
    def __init__(self, channel, high, gama=2, b=1):
        super(MECA_layer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gama))
        k_size = t if t % 2 else t + 1

        self.gamma = nn.Parameter(torch.zeros(1))

        # global
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_g = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # local
        self.conv_l = nn.Conv1d(high * high, high * high, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        batch, c, h, w = x.size()

        # global
        x_g = self.avg_pool(x)
        x_g = self.conv_g(x_g.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # local
        x_l = x.reshape(batch, c, -1)
        x_l = self.conv_l(x_l.transpose(-1, -2)).transpose(-1, -2)
        x_l = x_l.view(batch, c, h, w)

        x_lg = 2 * (1-self.sigmoid(self.gamma)) * x_l + 2 * self.sigmoid(self.gamma) * x_g
        # x_lg = x_l + x_g
        
        weight = self.sigmoid(x_lg)
        return x * weight


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        att = self.sigmoid(x)
        return att * x


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class channel_attention(nn.Module):

    def __init__(self, in_channel, ratio=4):

        super(channel_attention, self).__init__()


        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)


        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)


        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        b, c, h, w = inputs.shape


        max_pool = self.max_pool(inputs)

        avg_pool = self.avg_pool(inputs)


        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])


        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)


        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)


        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)


        x = x_maxpool + x_avgpool

        x = self.sigmoid(x)

        x = x.view([b, c, 1, 1])

        outputs = inputs * x

        return outputs


class spatial_attention(nn.Module):

    def __init__(self, kernel_size=7):

        super(spatial_attention, self).__init__()


        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)


        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)

        x = torch.cat([x_maxpool, x_avgpool], dim=1)


        x = self.conv(x)

        x = self.sigmoid(x)

        outputs = inputs * x

        return outputs


class cbam(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super(cbam, self).__init__()
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)


    def forward(self, inputs):

        x = self.channel_attention(inputs)

        x = self.spatial_attention(x)

        return x


class JK_att(nn.Module):
    def __init__(self, channel, layer=4, ratio=2):
        super(JK_att, self).__init__()
        hidden = channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.trans = nn.Linear(channel * layer, hidden)
        self.att = nn.Linear(channel + hidden, 1)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def forward(self, feature_list):
        batch_num = feature_list[0].shape[0]
        pool_list = []
        for feature in feature_list:
            pool_list.append(self.avg_pool(feature).view(batch_num, -1))
        # total_feature = sum(pool_list)
        total_feature = torch.cat(pool_list, dim=-1)
        # total_feature = sum(pool_list) / len(pool_list)
        ref_feature = self.relu(self.trans(total_feature))
        attention_scores = []
        for pool_feature in pool_list:
            attention_scores.append(self.relu(self.att(torch.cat([ref_feature, pool_feature], dim=1))))
        attention_scores = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
        return attention_scores

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.att.weight, gain=gain)
        nn.init.zeros_(self.att.bias)
        nn.init.xavier_uniform_(self.trans.weight, gain=gain)
        nn.init.zeros_(self.trans.bias)



class MCS_att(nn.Module):
    def __init__(self, channel, groups=4, gama=2, b=1):
        super(MCS_att, self).__init__()
        # t = int(abs((math.log(channel, 2) + b) / gama))
        # k_size = t if t % 2 else t + 1

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # dw
        # self.conv_d = nn.Conv2d(channel // groups, channel//groups, kernel_size=3, padding=1, groups=channel//groups, bias=False)
        # self.bn_d = nn.BatchNorm2d(channel // groups)

        # channel_att
        self.conv_c = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # spatial_att
        self.conv_h = nn.Conv2d(channel // (2 * groups), 1, 1, bias=False)
        self.conv_w = nn.Conv2d(channel // (2 * groups), 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        # group
        x = x.reshape(b * self.groups, -1, h, w)

        # dw卷积
        # x = self.relu(self.bn_d(self.conv_d(x)))

        # channel_attention
        x_c = self.avg_pool(x)
        x_c = self.conv_c(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = x * self.sigmoid(x_c)

        # split
        x_0, x_1 = x.chunk(2, dim=1)

        # attention_h
        x_h = self.pool_h(self.conv_h(x_0))
        x_h = self.sigmoid(x_h) * x_0

        # attention_w
        x_w = self.pool_w(self.conv_h(x_1))
        x_w = self.sigmoid(x_w) * x_1

        # concatenate along channel axis
        out = torch.cat([x_h, x_w], dim=1)

        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out



