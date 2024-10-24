import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.ResNet import BasicBlock


class AHBF(nn.Module):
    def __init__(self, inchannel):
        super(AHBF, self).__init__()
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel, 1,stride=1)
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)
        self.bn_v1 = nn.BatchNorm1d(2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x, y, logitx, logity):
        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit


class AHBF_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100,  num_branch=3, num_div=2):
        super(AHBF_ResNet, self).__init__()
        if block == BasicBlock:
            depth = sum(num_blocks) * 2 + 2
        else:
            depth = sum(num_blocks) * 3 + 2
        print('| ResNet-%d' % (depth))

        self.inplanes = 64
        self.num_branches = num_branch

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        fix_inplanes = self.inplanes
        for i in range(num_branch):
            setattr(self, 'layer4_' + str(i), self._make_layer(block, 512, num_blocks[3] + i * num_div, stride=2))
            self.inplanes = fix_inplanes
            setattr(self, 'classifier4_' + str(i), nn.Linear(512 * block.expansion, num_classes))

        for i in range(num_branch - 1):
            setattr(self, 'afm_' + str(i), AHBF(512 * block.expansion))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        logits_list = []
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # backbone
        out_0 = getattr(self, 'layer4_0')(x)
        feature_list.append(out_0)
        out_0 = self.avg_pool(out_0)
        out_0 = torch.flatten(out_0, 1)
        out_0 = getattr(self, 'classifier4_0')(out_0)
        logits_list.append(out_0)

        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer4_'+str(i))(x)
            feature_list.append(temp)
            temp = self.avg_pool(temp)
            temp = torch.flatten(temp, 1)
            temp = getattr(self, 'classifier4_'+str(i))(temp)
            logits_list.append(temp)

        ensem_fea = []
        ensem_logits = []

        for i in range(0, self.num_branches - 1):
            if i == 0:
                ensembleff, logit = getattr(self, 'afm_' + str(i))(feature_list[i], feature_list[i + 1], logits_list[i],
                                                                   logits_list[i + 1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff, logit = getattr(self, 'afm_' + str(i))(ensem_fea[i - 1], feature_list[i + 1],
                                                                   ensem_logits[i - 1], logits_list[i + 1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
        return logits_list, ensem_logits


def AHBF_ResNet18(num_classes, num_branches=3, aux=2):
    return AHBF_ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes, num_branch=num_branches,num_div=aux)


if __name__ == '__main__':
    from thop import profile, clever_format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AHBF_ResNet18(num_classes=100, num_branches=3).cuda()
    print(model)
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
