import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FuzzyUNet1.FuzzyLayer import FuzzyLayer


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


#####################################################################################################
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.fuzzy_1 = FuzzyLayer(fuzzynum=1, channel=nb_filter[2])
        self.fbn1 = nn.BatchNorm2d(nb_filter[2], affine=True)
        self.fuzzy_2 = FuzzyLayer(fuzzynum=1, channel=nb_filter[3])
        self.fbn2 = nn.BatchNorm2d(nb_filter[3], affine=True)
        self.fuzzy_3 = FuzzyLayer(fuzzynum=1, channel=nb_filter[4])
        self.fbn3 = nn.BatchNorm2d(nb_filter[4], affine=True)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.fbn1(self.fuzzy_1(x2_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.fbn2(self.fuzzy_2(x3_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.fbn3(self.fuzzy_3(x4_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 示例使用
input_tensor = torch.randn(1, 3, 512, 512)  # 示例输入张量，假设通道数为3，图像大小为256x256
input_tensor = input_tensor.to(device)

# 创建模型
net = UNet(2)
net = net.to(device)

result = net(input_tensor)
print(result.shape)

