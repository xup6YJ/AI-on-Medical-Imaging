# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

kx = ky = kz = 3
px = py = pz = 1
epsilon = 1e-7

norm_mode = 'bn'
num_groups = None
ch_per_group = 16


class Norm(nn.Module):
    def __init__(self, channel):
        super(Norm, self).__init__()
        if norm_mode == 'bn':
            self.norm = nn.BatchNorm3d(channel)
        elif norm_mode == 'gn':
            if num_groups is not None and ch_per_group is not None:
                raise ValueError('Can only choose one, num_groups or ch_per_group')
            if num_groups is not None:
                assert channel%num_groups == 0, 'channel%%num_groups != 0'
                self.norm = nn.GroupNorm(num_groups, channel)
            elif ch_per_group is not None:
                assert channel%ch_per_group == 0, 'channel%%ch_per_group != 0'
                self.norm = nn.GroupNorm(channel//ch_per_group, channel)
            else:
                raise ValueError('Please choose one, num_groups or ch_per_group')
        else:
            raise ValueError('Unknown normalization mode')

    def forward(self, x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = (scale, scale, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2):
        super(DeConvBlock, self).__init__()
        k = (scale, scale, 1)
        s = (scale, scale, 1)
        self.deconv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=k, stride=s, padding=0)
    def forward(self, x):
        x = self.deconv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, deconv=False):
        super(UpConvBlock, self).__init__()
        if deconv:
            self.up = nn.Sequential(
                DeConvBlock(in_channel, out_channel, scale))
        else:
            layers = [Upsample(scale)]
            if in_channel != out_channel:
                layers.append(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0))
            self.up = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(ConvBlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(in_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, channel, bias=True, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=k, stride=1, padding=p, bias=bias),
            Norm(channel),
			nn.ReLU(inplace=True))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1


# Recurrent CNN
class RCNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, num_rcnn=2, t=2):
        super(RCNNBlock, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(num_rcnn):
            layers.append(RecurrentBlock(out_channel, bias=bias, t=t))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.nn(out)
        return out


# Recurrent Residual CNN
class RRCNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, num_rcnn=2, t=2):
        super(RRCNNBlock, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(num_rcnn):
            layers.append(RecurrentBlock(out_channel, bias=bias, t=t))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.nn(x1)
        return x1 + x2


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, bias=True):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(f_int))
        self.w_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(f_int))
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x * psi


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class BAMChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(BAMChannelGate, self).__init__()
        gate_c = [Flatten()]
        gate_channels = [gate_channel]
        gate_channels += [gate_channel//reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels)-2):
            gate_c += [
                nn.Linear(gate_channels[i], gate_channels[i+1]),
                nn.BatchNorm1d(gate_channels[i+1]),
                nn.ReLU(inplace=True)
            ]
        gate_c.append(nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.gate_c = nn.Sequential(*gate_c)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        att = self.gate_c(avg_pool)
        att = att.reshape(att.shape[0], att.shape[1], 1, 1, 1).expand_as(x)
        return att


class BAMSpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(BAMSpatialGate, self).__init__()
        gate_s = [
            nn.Conv3d(gate_channel, gate_channel//reduction_ratio, kernel_size=1),
            nn.BatchNorm3d(gate_channel//reduction_ratio),
            nn.ReLU(inplace=True)
        ]
        for _ in range(dilation_conv_num):
            gate_s += [
                nn.Conv3d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, padding=dilation_val, dilation=dilation_val),
                nn.BatchNorm3d(gate_channel//reduction_ratio),
                nn.ReLU(inplace=True)
            ]
        gate_s.append(nn.Conv3d(gate_channel//reduction_ratio, 1, kernel_size=1))
        self.gate_s = nn.Sequential(*gate_s)

    def forward(self, x):
        att = self.gate_s(x).expand_as(x)
        return att


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = BAMChannelGate(gate_channel)
        self.spatial_att = BAMSpatialGate(gate_channel)

    def forward(self, x):
        att_c = self.channel_att(x)
        att_s = self.spatial_att(x)
        scale = 1 + torch.sigmoid(att_c+att_s)
        return x * scale


class BAMAPBlock(nn.Module):
    def __init__(self, channel):
        super(BAMAPBlock, self).__init__()
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.bam = BAM(channel)

    def forward(self, x1, x2):
        out = self.conv(torch.cat((x1, x2), dim=1))
        out = self.bam(out)
        return out


# Classification Guided Module
class CGM(nn.Module):
    def __init__(self, in_channel):
        super(CGM, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channel, 1, kernel_size=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((50, 50, 1)))
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(2500, 2))

    def forward(self, x):
        out = self.net(x)
        out = self.classifier(out)
        return out


class ResConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(ResConvBlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(out_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv(x1)
        return x1 + x2
