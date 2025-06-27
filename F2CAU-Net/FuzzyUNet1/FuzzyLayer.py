import itertools

import numpy as np
import torch
from torch import nn


class FuzzyLayerOriginal(nn.Module):
    def __init__(self, fuzzynum, channel):
        super(FuzzyLayerOriginal, self).__init__()
        self.n = fuzzynum
        self.channel = channel
        self.conv1 = nn.Conv2d(self.channel, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, self.channel, 3, padding=1)
        self.mu = nn.Parameter(torch.randn((self.channel, self.n)))
        self.sigma = nn.Parameter(torch.randn((self.channel, self.n)))
        self.bn1 = nn.BatchNorm2d(1, affine=True)
        self.bn2 = nn.BatchNorm2d(self.channel, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        tmp = torch.tensor(np.zeros((x.size()[0], x.size()[1], x.size()[2], x.size()[3])), dtype=torch.float).cuda()
        for num, channel, w, h in itertools.product(range(x.size()[0]), range(x.size()[1]), range(x.size()[2]),
                                                    range(x.size()[3])):
            for f in range(self.n):
                tmp[num][channel][w][h] -= ((x[num][channel][w][h] - self.mu[channel][f]) / self.sigma[channel][f]) ** 2
        fNeural = self.bn2(self.conv2(self.bn1(torch.exp(tmp))))
        return fNeural


class FuzzyLayer(nn.Module):
    def __init__(self, fuzzynum, channel):
        super(FuzzyLayer, self).__init__()
        self.n = fuzzynum
        self.channel = channel
        self.conv1 = nn.Conv2d(self.channel, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, self.channel, 3, padding=1)
        self.mu = nn.Parameter(torch.randn((self.channel, self.n)))
        self.sigma = nn.Parameter(torch.randn((self.channel, self.n)))
        self.bn1 = nn.BatchNorm2d(1, affine=True)
        self.bn2 = nn.BatchNorm2d(self.channel, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        num_samples, num_channels, height, width = x.size()

        x = x.view(num_samples, num_channels, -1)  # Reshape to (N, C, H * W)

        mu = self.mu.view(self.channel, self.n, 1)  # (C, F, 1)
        sigma = self.sigma.view(self.channel, self.n, 1)  # (C, F, 1)

        x_normalized = (x.unsqueeze(2) - mu) / sigma  # (N, C, F, H * W)
        x_normalized = x_normalized.view(num_samples, -1, height, width)  # Reshape to (N, C * F, H, W)

        tmp = torch.sum(-x_normalized ** 2, dim=1)  # Sum along the channel dimension
        tmp = tmp.unsqueeze(1)  # Add a channel dimension

        fNeural = self.bn2(self.conv2(self.bn1(torch.exp(tmp))))
        return fNeural