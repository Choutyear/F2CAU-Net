import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyMembershipFunction(nn.Module):
    def __init__(self, a, b, c, d):
        super(FuzzyMembershipFunction, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        return torch.max(torch.min((x - self.a) / (self.b - self.a), (self.d - x) / (self.d - self.c)), torch.tensor(0.))


class FuzzyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FuzzyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.fmf = FuzzyMembershipFunction(0., 0.3, 0.7, 1.)  # This is an example, you can modify it to fit your needs

    def forward(self, x):
        conv_output = self.conv(x)
        return self.fmf(conv_output)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            FuzzyConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            FuzzyConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FuzzyUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(FuzzyUNet, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# # 创建模型实例
# num_classes = 1  # 在二值分割任务中通常使用1个输出通道
# input_channels = 3  # 输入图像通道数
# model = FuzzyUNet(num_classes, input_channels)
