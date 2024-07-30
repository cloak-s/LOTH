import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Basic_aux(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Basic_aux, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(out)
        return out


class Bottle_aux(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2, Groups=False, factor=1):
        super(Bottle_aux, self).__init__()
        self.group_conv = Groups
        mid_plane = planes // factor
        self.conv1 = nn.Conv2d(in_planes, mid_plane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_plane)

        if self.group_conv:
            self.conv2 = nn.Conv2d(
                mid_plane, mid_plane, kernel_size=3, stride=stride, padding=1, bias=False, groups=mid_plane
            )
        else:
            self.conv2 = nn.Conv2d(
                mid_plane, mid_plane, kernel_size=3, stride=stride, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(mid_plane)
        self.conv3 = nn.Conv2d(
            mid_plane, planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class AGB_aux(nn.Module):  # DBFSKD
    expansion = 1

    def __init__(self, in_planes, planes, stride=2):
        super(AGB_aux, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, in_planes, stride=stride, kernel_size=3, padding=1, bias=False,
                               groups=in_planes)
        self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=planes
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = F.relu(x)
        out = F.relu(self.bn0(self.conv0(x)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class DSC_aux(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSC_aux, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        out = F.relu(inputs)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return out


class LCT_aux(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(LCT_aux, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = self.bn(self.conv(x))
        return x


class SCAN_aux(nn.Module):
    def __init__(self, channel_in, channel_out, size):
        super(SCAN_aux, self).__init__()

        self.trans = nn.Sequential(
            nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=size, stride=size),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.trans(x)
        return x


