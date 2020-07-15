import torch
import torch.nn as nn
from torch.nn import functional as F


class DirectKernelEstimator(nn.Module):
    def __init__(self, nf):
        super(DirectKernelEstimator, self).__init__()
        # [64, 128, 128]
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv0 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(nf, 3, 1, 1, bias=True)

    def forward(self, x):
        """
        Forward function of classifier
        :param x: concatenated input
        :return:
        """
        fea = self.lrelu(self.conv0(x))
        fea = self.lrelu(self.conv1(fea))
        fea = self.lrelu(self.conv2(fea))
        fea = self.lrelu(self.conv3(fea))
        fea = self.lrelu(self.conv4(fea))
        fea = self.lrelu(self.conv5(fea))
        fea = self.conv6(fea)

        small_x = F.avg_pool2d(x, 2)
        out = fea + small_x
        return out


class DirectKernelEstimator_CMS(nn.Module):
    def __init__(self, nf):
        super(DirectKernelEstimator_CMS, self).__init__()
        # [64, 128, 128]
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pad = nn.ReflectionPad2d(1)
        self.conv0 = nn.Conv2d(3, nf, 3, 1, 0, bias=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(nf, nf * 2, 4, 2, padding=0, bias=True)
        self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(nf * 2, nf, 3, 1, padding=0, bias=True)
        self.conv6 = nn.Conv2d(nf, 3, 1, stride=1, padding=0, bias=True)
    def forward(self, x):
        """
        Forward function of classifier
        :param x: concatenated input
        :return:
        """
        m = x.mean(2, keepdim=True).mean(3, keepdim=True)
        fea = self.lrelu(self.conv0(self.pad(x - m)))
        fea = self.lrelu(self.conv1(self.pad(fea)))
        fea = self.lrelu(self.conv2(self.pad(fea)))
        fea = self.lrelu(self.conv3(self.pad(fea)))
        fea = self.lrelu(self.conv4(self.pad(fea)))
        fea = self.lrelu(self.conv5(self.pad(fea)))
        fea = self.conv6(fea)
        #small_x = F.avg_pool2d(x, 2)
        out = fea + m #small_x
        return out


class DirectKernelEstimatorVideo(nn.Module):
    def __init__(self, nf, in_nc=3, scale=2):
        super(DirectKernelEstimatorVideo, self).__init__()
        # [64, 128, 128]
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pad3d = nn.ReplicationPad3d(1)
        self.pad = nn.ReflectionPad2d(1)
        self.conv0 = nn.Conv3d(in_nc, nf, 3, 1, 0, bias=True)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(nf, nf * 2, 4, 2, 0, bias=True)
        if scale == 2:
            self.conv3 = nn.Conv2d(nf * 2, nf, 3, 1, 0, bias=True)
        elif scale == 4:
            self.conv3 = nn.Conv2d(nf * 2, nf, 4, 2, 0, bias=True)
        else:
            raise NotImplementedError()
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 0, bias=True)
        self.conv5 = nn.Conv3d(nf, nf, 3, 1, 0, bias=True)
        self.conv6 = nn.Conv2d(nf, in_nc, 1, 1, 0, bias=True)

        self.scale = scale

    def forward(self, x):
        """
        Forward function of classifier
        :param x: B C T H W
        :return:
        """
        B, C, T, H, W = x.shape
        m = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        x = x - m
        # No nn.Reflectionpad3d
        x = self.lrelu(self.conv0(self.pad3d(x)))
        # B C T H W  -> B*T C H W
        fea = x.transpose(1, 2).reshape(B*T, -1, H, W)
        fea = self.lrelu(self.conv1(self.pad(fea)))
        fea = self.lrelu(self.conv2(self.pad(fea)))
        fea = self.lrelu(self.conv3(self.pad(fea)))
        fea = self.lrelu(self.conv4(self.pad(fea)))
        # B*T C H W -> B C T H W
        fea = fea.reshape(B, T, -1, H//self.scale, W//self.scale).transpose(1, 2)
        fea = self.lrelu(self.conv5(self.pad3d(fea)))
        # B C T H W  -> B*T C H W
        fea = fea.transpose(1, 2).reshape(B * T, -1, H//self.scale, W//self.scale)
        fea = self.conv6(fea)
        fea = fea.reshape(B, T, -1, H//self.scale, W//self.scale).transpose(1, 2)
        out = fea + m
        return out
