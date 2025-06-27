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
    def __init__(self):
        super(FuzzyUNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(1, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)

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
        x1 = self.down_conv_1(image)  #
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)

        # decoder
        x = self.up_trans_1(x7)
        # Ensure the number of channels is 512 after the feature fusion
        y = crop_img(x3, x)
        y = F.pad(y, (0, x.size(3) - y.size(3), 0, x.size(2) - y.size(2)))  # Zero-padding to match size
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.up_trans_2(x)
        # Ensure the number of channels is 128 after the feature fusion
        y = crop_img(x2, x)
        y = F.pad(y, (0, x.size(3) - y.size(3), 0, x.size(2) - y.size(2)))  # Zero-padding to match size
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.up_trans_3(x)
        # Ensure the number of channels is 64 after the feature fusion
        y = crop_img(x1, x)
        y = F.pad(y, (0, x.size(3) - y.size(3), 0, x.size(2) - y.size(2)))  # Zero-padding to match size
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.out(x)
        return x


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


flag = False

if flag:
    model = FuzzyUNet()
    print(model)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(output)
