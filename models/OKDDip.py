import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch
from thop import clever_format, profile
from models.ResNet import BasicBlock


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)  # 注意，这里没有detach掉
        return loss


class OKDDip_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, factor=8, num_branch=3):
        super(OKDDip_ResNet, self).__init__()
        depth = sum(num_blocks) * 2 + 2
        print('| OKDD_ResNet-%d' % depth)
        self.inplanes = 64
        self.num_branch = num_branch
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        fix_planes = self.inplanes

        for i in range(num_branch):  # 最后一个默认为 group——leader
            setattr(self, 'layer4_'+str(i), self._make_layer(block, 512, num_blocks[3], stride=2))
            self.inplanes = fix_planes
            setattr(self, 'classifier_'+str(i), nn.Linear(512 * block.expansion, num_classes))

        self.query_weight = nn.Linear(512, 512//factor, bias=False)
        self.key_weight = nn.Linear(512, 512//factor, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        proj_q = []
        proj_k = []
        s_logits = []

        # 第一个分支
        branch_0 = getattr(self, 'layer4_0')(x)
        branch_0 = self.avg_pool(branch_0)
        branch_0 = torch.flatten(branch_0, 1)

        # att
        q1_weight = self.query_weight(branch_0).unsqueeze(dim=1)
        k1_weight = self.key_weight(branch_0).unsqueeze(dim=1)
        proj_q.append(q1_weight)
        proj_k.append(k1_weight)

        # logits
        branch_0 = getattr(self, 'classifier_0')(branch_0)
        s_logits.append(branch_0.unsqueeze(dim=-1))

        for i in range(1, self.num_branch-1):
            temp = getattr(self, 'layer4_'+str(i))(x)
            temp = self.avg_pool(temp)
            temp = torch.flatten(temp, 1)
            proj_q.append(self.query_weight(temp).unsqueeze(dim=1))
            proj_k.append(self.key_weight(temp).unsqueeze(dim=1))
            temp_out = getattr(self, 'classifier_'+str(i))(temp)
            s_logits.append(temp_out.unsqueeze(dim=-1))


        # 注意力机制
        proj_q = torch.cat(proj_q, dim=1)
        proj_k = torch.cat(proj_k, dim=1)
        energy = torch.bmm(proj_q, proj_k.permute(0, 2, 1))
        attention = F.softmax(energy, dim=-1)
        a_branch = torch.bmm(torch.cat(s_logits, dim=-1), attention.permute(0, 2, 1))

        # group_leader
        branch_gl = getattr(self, 'layer4_'+str(self.num_branch-1))(x)
        branch_gl = self.avg_pool(branch_gl)
        branch_gl = torch.flatten(branch_gl, 1)
        branch_gl = getattr(self, 'classifier_'+str(self.num_branch-1))(branch_gl)

        a_branch = [a_branch[:, :, i] for i in range(len(s_logits))]
        s_logits = [logit.squeeze() for logit in s_logits]
        return s_logits, a_branch, branch_gl


def OKDDip_Res_18(num_classes, factor, num_branch=4):
    return OKDDip_ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes, factor=factor, num_branch=num_branch)



if __name__ == '__main__':
    import torch
    from thop import profile, clever_format

    x = torch.randn([5, 3, 32, 32])

    model = OKDDip_Res_18(100, factor=8, num_branch=3)
    print(model)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)