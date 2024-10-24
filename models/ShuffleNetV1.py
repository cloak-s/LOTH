import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Auxiliary import Bottle_aux, DSC_aux


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)


class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups, before_relu=False):
        super().__init__()
        self.before_relu = before_relu
        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv2d(
                input_channels,
                int(output_channels / 4),
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv2d(
                    input_channels,
                    int(output_channels / 4),
                    groups=1
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU()
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 × 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )
            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)  # 通道 shuffle
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)   # 逐点卷积

        output = self.fusion(shortcut, shuffled)
        if self.before_relu:
            return output
        else :
            return F.relu(output)



class ShuffleNet(nn.Module):
    def __init__(self,  num_classes=100, groups=3, aux=None):
        super().__init__()

        print('| ShuffleNetV1-{}'.format(aux))
        num_blocks = [4, 8, 4]

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]
        # self.before_act = before_act

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)
        self.aux = aux

        if self.aux == 'G-Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 240, 2, Groups=True),
                Bottle_aux(240, 480, 2, Groups=True),
                Bottle_aux(480, 960, 2, Groups=True),
            )
            self.linear_1 = nn.Linear(960, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(240, 480, 2, Groups=True),
                Bottle_aux(480, 960, 2, Groups=True),
            )
            self.linear_2 = nn.Linear(960, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(480, 960, 2, Groups=True),
            )
            self.linear_3 = nn.Linear(960, num_classes)

        elif self.aux == 'Bottle':
            self.aux_1_trans = nn.Sequential(
                Bottle_aux(24, 240, 2),
                Bottle_aux(240, 480, 2),
                Bottle_aux(480, 960, 2),
            )
            self.linear_1 = nn.Linear(960, num_classes)

            self.aux_2_trans = nn.Sequential(
                Bottle_aux(240, 480, 2),
                Bottle_aux(480, 960, 2),
            )
            self.linear_2 = nn.Linear(960, num_classes)

            self.aux_3_trans = nn.Sequential(
                Bottle_aux(480, 960, 2),
            )
            self.linear_3 = nn.Linear(960, num_classes)

        elif self.aux == 'DSC':
            self.aux_1_trans = nn.Sequential(
                DSC_aux(24, 240, 2),
                DSC_aux(240, 480, 2),
                DSC_aux(480, 960, 2),
            )
            self.linear_1 = nn.Linear(960, num_classes)

            self.aux_2_trans = nn.Sequential(
                DSC_aux(240, 480, 2),
                DSC_aux(480, 960, 2),
            )
            self.linear_2 = nn.Linear(960, num_classes)

            self.aux_3_trans = nn.Sequential(
                DSC_aux(480, 960, 2),
            )
            self.linear_3 = nn.Linear(960, num_classes)

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
        x = self.conv1(x)
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
        emb4 = x
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.aux:
            return [emb1, emb2, emb3, emb4], [out_1, out_2, out_3, out]
        else:
            return emb4, out

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups, before_relu=False):
        """make shufflenet stage

        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups,
                    before_relu=before_relu
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)




if __name__ == "__main__":

    # set groups to 60
    from thop import profile, clever_format
    x = torch.randn([1, 3, 32, 32])

    shuffleNet = ShuffleNet(aux=None, num_classes=200)
    print(shuffleNet)
    flops, params = profile(shuffleNet, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

