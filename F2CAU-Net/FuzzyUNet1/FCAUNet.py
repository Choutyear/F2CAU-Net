import math

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


# 自定义高斯模糊卷积层
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        # 生成高斯卷积核
        kernel = self._gaussian_kernel(self.kernel_size, self.sigma).to(x.device)

        # 执行卷积操作
        x = F.conv2d(x, kernel, padding=self.kernel_size // 2)  # padding保持输出大小与输入相同

        return x

    def _gaussian_kernel(self, size, sigma):
        kernel = torch.tensor([[math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
                                for x in range(size)]], dtype=torch.float32)
        kernel = kernel / kernel.sum()  # 归一化
        return kernel


class FuzzyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FuzzyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.fmf = FuzzyMembershipFunction(0., 1.0, 0., 1.)  # This is an example, you can modify it to fit your needs
        self.fmf = GaussianBlur(5, 1.0)  # This is an example, you can modify it to fit your needs

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


# class FuzzySelfAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim=None):
#         super(FuzzySelfAttention, self).__init__()
#
#         if hidden_dim is None:
#             hidden_dim = input_dim // 2
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.query = nn.Linear(input_dim, hidden_dim)
#         self.key = nn.Linear(input_dim, hidden_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.fmf = FuzzyMembershipFunction(0., 0.3, 0.7, 1.)  # This is an example, you can modify it to fit your needs
#
#     def forward(self, x):
#         x = torch.squeeze(x)
#         batch_size, seq_len, _ = x.size()
#
#         # Query, Key, and Value
#         q = self.query(x)  # (batch_size, seq_len, hidden_dim)
#         k = self.key(x)  # (batch_size, seq_len, hidden_dim)
#         v = self.value(x)  # (batch_size, seq_len, input_dim)
#
#         # Attention scores
#         attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, seq_len)
#         attn_scores = attn_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
#
#         # Attention weights
#         attn_weights = torch.softmax(attn_scores, dim=-1)
#         attn_weights = self.fmf(attn_weights)
#
#         # Attention result
#         output = torch.bmm(attn_weights, v)  # (batch_size, seq_len, input_dim)
#
#         return output


class FuzzySelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(FuzzySelfAttention, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        # self.fmf = FuzzyMembershipFunction(0., 1.0, 0., 1.)  # This is an example, you can modify it to fit your needs
        self.fmf = GaussianBlur(5, 1.0)

    def forward(self, x):
        batch_size, num_channels, seq_len, _ = x.size()
        x = x.view(batch_size * num_channels, seq_len, -1)  # Reshape to (batch_size * num_channels, seq_len, input_dim)

        # Query, Key, and Value
        q = self.query(x)  # (batch_size * num_channels, seq_len, hidden_dim)
        k = self.key(x)  # (batch_size * num_channels, seq_len, hidden_dim)
        v = self.value(x)  # (batch_size * num_channels, seq_len, input_dim)

        # Attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size * num_channels, seq_len, seq_len)
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        # Attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.fmf(attn_weights)

        # Attention result
        output = torch.bmm(attn_weights, v)  # (batch_size * num_channels, seq_len, input_dim)
        output = output.view(batch_size, num_channels, seq_len, -1)  # Reshape back to (batch_size, num_channels, seq_len, input_dim)

        return output


class FCAUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(FCAUNet, self).__init__()

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

        self.fuzzy_attn1 = FuzzySelfAttention(nb_filter[1])
        self.fuzzy_attn2 = FuzzySelfAttention(nb_filter[2])
        self.fuzzy_attn3 = FuzzySelfAttention(nb_filter[3])
        self.fuzzy_attn4 = FuzzySelfAttention(nb_filter[4])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.fuzzy_attn1(x3_1)  # apply fuzzy attention
        # x3_1 = torch.unsqueeze(x3_1, 0)

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.fuzzy_attn2(x2_2)  # apply fuzzy attention
        # x2_2 = torch.unsqueeze(x2_2, 0)

        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3 = self.fuzzy_attn3(x1_3)  # apply fuzzy attention
        # x1_3 = torch.unsqueeze(x1_3, 0)

        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.fuzzy_attn4(x0_4)  # apply fuzzy attention
        # x0_4 = torch.unsqueeze(x0_4, 0)

        output = self.final(x0_4)
        return output


# # 创建模型实例
# num_classes = 1  # 在二值分割任务中通常使用1个输出通道
# input_channels = 3  # 输入图像通道数
# model = FuzzyUNet(num_classes, input_channels)
