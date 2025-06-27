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


class FuzzySelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(FuzzySelfAttention, self).__init__()
        self.attention_size = attention_size
        self.query_layer = nn.Linear(attention_size, attention_size)
        self.key_layer = nn.Linear(attention_size, attention_size)
        self.fmf = FuzzyMembershipFunction(0., 0.3, 0.7, 1.)  # This is an example, you can modify it to fit your needs

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)

        attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)
        attn_weights = self.fmf(attn_weights)  # apply fuzzy logic to the attention weights

        return torch.bmm(attn_weights, x)


class UNetWithFuzzyAttn(nn.Module):
    def __init__(self, attention_size=64):
        super(UNetWithFuzzyAttn, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(1, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)

        self.fuzzy_attn1 = FuzzySelfAttention(attention_size)
        self.fuzzy_attn2 = FuzzySelfAttention(attention_size)
        self.fuzzy_attn3 = FuzzySelfAttention(attention_size)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)

        self.up_conv_1 = DoubleConv(512, 256)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)

        self.up_conv_2 = DoubleConv(256, 128)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)

        self.up_conv_3 = DoubleConv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1)

    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)

        # decoder
        x = self.up_trans_1(x7)
        y = crop_img(x3, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.fuzzy_attn1(x)  # apply fuzzy attention

        x = self.up_trans_2(x)
        y = crop_img(x2, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.fuzzy_attn2(x)  # apply fuzzy attention

        x = self.up_trans_3(x)
        y = crop_img(x1, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        x = self.fuzzy_attn3(x)  # apply fuzzy attention

        x = self.out(x)
        return x


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]