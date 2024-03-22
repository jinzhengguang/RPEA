"""
From https://github.com/mit-han-lab/e3d/blob/db65d6c968a12d30a4caa2715ef6766ec04d7505/spvnas/core/models/semantic_kitti/minkunet.py
"""

import time
from collections import OrderedDict
import torch
import torchsparse
import torch.nn as nn
import torchsparse.nn as spnn

__all__ = ['MinkUNet']

# _ks = 7
_ks = 3


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 transpose=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 2023-03-21 Jinzheng Guang CBAM attention
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spnnavg = spnn.GlobalAveragePooling()
        self.spnnmax = spnn.GlobalMaxPooling()

    def forward(self, x):
        xtemp = x.F.contiguous().permute(1, 0)
        # maxp = self.spnnmax(x)
        avgp = self.spnnavg(x)
        max_result = self.maxpool(xtemp)
        avg_result = self.avgpool(avgp.permute(1,0))
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        outputs = xtemp * output
        x.F = outputs.permute(1, 0)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        if stride == 1 and ks == 3:
            ks = _ks
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc)
        )
        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )
        self.relu = spnn.ReLU(True)
        self.catt = ChannelAttention(channel=outc)
        self.att_status = False

    def forward(self, x):
        x1 = self.net(x)
        # xtemp = x1.F.contiguous()
        if self.att_status:
            x1 = self.catt(x1)
        out = self.relu(x1 + self.downsample(x))
        # out = self.relu(self.net(x) + self.downsample(x))
        return out
class ResidualPath(nn.Module):
    def __init__(self, inc, outc, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=3, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
        )
        self.downsample = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=1, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc)
        )
        self.relu = spnn.ReLU(True)
        self.basicconv = BasicConvolutionBlock(inc, outc, ks=3)
        self.catt = ChannelAttention(channel=outc)
        self.path_status = True

    def forward(self, x):
        out = x
        if self.path_status:
            # out = self.relu(self.net(x) + self.downsample(x))
            out = self.relu(self.catt(self.basicconv(x)) + x)
        return out

class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            # spnn.Conv3d(3, cs[0], kernel_size=3, stride=1),
            spnn.Conv3d(input_dim, cs[0], kernel_size=_ks, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            # spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.Conv3d(cs[0], cs[0], kernel_size=_ks, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])
        # 2023-03-31 Jinzheng Guang
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.pathx0 = nn.Sequential(
            ResidualPath(cs[0], cs[0]),
            ResidualPath(cs[0], cs[0]),
            ResidualPath(cs[0], cs[0]),
            ResidualPath(cs[0], cs[0])
        )
        self.pathx1 = nn.Sequential(
            ResidualPath(cs[1], cs[1]),
            ResidualPath(cs[1], cs[1]),
            ResidualPath(cs[1], cs[1])
        )
        self.pathx2 = nn.Sequential(
            ResidualPath(cs[2], cs[2]),
            ResidualPath(cs[2], cs[2])
        )
        self.pathx3 = nn.Sequential(
            ResidualPath(cs[3], cs[3])
        )

        self.classifier = nn.Sequential(nn.Linear(cs[8],
                                                  kwargs['num_classes']))


        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)  # 621,687 x 3
        x1 = self.stage1(x0)  # 621,687 x 32
        x2 = self.stage2(x1)  # 362,687 x 32
        x3 = self.stage3(x2)  # 192,434 x 64
        x4 = self.stage4(x3)  # 94,584 x 128

        y1 = self.up1[0](x4)  # 42,187 x 256
        y1 = torchsparse.cat([y1, self.pathx3(x3)])
        y1 = self.up1[1](y1)  # 94,584 x 256

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, self.pathx2(x2)])
        y2 = self.up2[1](y2)  # 192,434 x 128

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, self.pathx1(x1)])
        y3 = self.up3[1](y3)  # 362,687 x 96

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, self.pathx0(x0)])
        y4 = self.up4[1](y4)  # 621,687 x 96

        out = self.classifier(y4.F)  # 621,687 x 31

        return out  # (n, 31)
