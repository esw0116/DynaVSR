import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class DirectKernelEstimator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, use_bn=True):
        super(DirectKernelEstimator, self).__init__()
        self.use_bn = use_bn
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        if use_bn:
            self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        if use_bn:
            self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=True)
        if use_bn:
            self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        if use_bn:
            self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        if use_bn:
            self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=True)
        if use_bn:
            self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=True)
        if use_bn:
            self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=True)
        if use_bn:
            self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=True)
        if use_bn:
            self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        # regression
        self.regressor = nn.Sequential(
            nn.Linear(nf * 8, out_nc * out_nc),
            nn.ReLU(inplace=True),
            nn.Linear(out_nc * out_nc, out_nc * out_nc)
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        """
        Forward function of classifier
        :param x: concatenated input
        :return:
        """
        if self.use_bn:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.bn0_1(self.conv0_1(fea)))

            fea = self.relu(self.bn1_0(self.conv1_0(fea)))
            fea = self.relu(self.bn1_1(self.conv1_1(fea)))

            fea = self.relu(self.bn2_0(self.conv2_0(fea)))
            fea = self.relu(self.bn2_1(self.conv2_1(fea)))

            fea = self.relu(self.bn3_0(self.conv3_0(fea)))
            fea = self.relu(self.bn3_1(self.conv3_1(fea)))

            fea = self.relu(self.bn4_0(self.conv4_0(fea)))
            fea = self.relu(self.bn4_1(self.conv4_1(fea)))
        else:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.conv0_1(fea))

            fea = self.relu(self.conv1_0(fea))
            fea = self.relu(self.conv1_1(fea))

            fea = self.relu(self.conv2_0(fea))
            fea = self.relu(self.conv2_1(fea))

            fea = self.relu(self.conv3_0(fea))
            fea = self.relu(self.conv3_1(fea))

            fea = self.relu(self.conv4_0(fea))
            fea = self.relu(self.conv4_1(fea))

        fea = nn.AdaptiveMaxPool2d(1)(fea)
        fea = fea.view(fea.size(0), -1)
        out = self.regressor(fea)
        out = out.reshape(-1, self.out_nc, self.out_nc)
        return fea, out


class GaussargsKernelEstimator(nn.Module):
    def __init__(self, in_nc, nf, use_bn=True):
        super(GaussargsKernelEstimator, self).__init__()
        self.use_bn = use_bn
        # Patch_size: 64x64
        # [nf, 64, 64]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        if use_bn:
            self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [2 * nf, 32, 32]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        if use_bn:
            self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=True)
        if use_bn:
            self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [4 * nf, 16, 16]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        if use_bn:
            self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        if use_bn:
            self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [8 * nf, 8, 8]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
        if use_bn:
            self.bn3_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        if use_bn:
            self.bn3_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # Regression
        self.regressor = nn.Sequential(
            nn.Linear(nf * 4, nf), nn.ReLU(inplace=True), nn.Linear(nf, 3))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward function of classifier
        :param x: concatenated input
        :return:
        """
        if self.use_bn:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.bn0_1(self.conv0_1(fea)))

            fea = self.relu(self.bn1_0(self.conv1_0(fea)))
            fea = self.relu(self.bn1_1(self.conv1_1(fea)))

            fea = self.relu(self.bn2_0(self.conv2_0(fea)))
            fea = self.relu(self.bn2_1(self.conv2_1(fea)))

            fea = self.relu(self.bn3_0(self.conv3_0(fea)))
            fea = self.relu(self.bn3_1(self.conv3_1(fea)))
        else:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.conv0_1(fea))

            fea = self.relu(self.conv1_0(fea))
            fea = self.relu(self.conv1_1(fea))

            fea = self.relu(self.conv2_0(fea))
            fea = self.relu(self.conv2_1(fea))

            fea = self.relu(self.conv3_0(fea))
            fea = self.relu(self.conv3_1(fea))

        fea = nn.AdaptiveAvgPool2d(1)(fea)
        fea = fea.view(fea.size(0), -1)
        out = self.regressor(fea)

        return fea, out


class AllargsKernelEstimator(nn.Module):
    def __init__(self, in_nc, nf, use_bn=True):
        super(AllargsKernelEstimator, self).__init__()
        self.use_bn = use_bn
        # Patch_size: 64x64
        # [nf, 64, 64]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        if use_bn:
            self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [2 * nf, 32, 32]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        if use_bn:
            self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=True)
        if use_bn:
            self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [4 * nf, 16, 16]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        if use_bn:
            self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        if use_bn:
            self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [8 * nf, 8, 8]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
        if use_bn:
            self.bn3_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        if use_bn:
            self.bn3_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(nf * 4, nf), nn.ReLU(inplace=True), nn.Linear(nf, 4)
        )
        # Regression
        self.regressor = nn.Sequential(
            nn.Linear(nf * 4, nf), nn.ReLU(inplace=True), nn.Linear(nf, 3))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward function of classifier
        :param x: concatenated input
        :return:
        """
        if self.use_bn:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.bn0_1(self.conv0_1(fea)))

            fea = self.relu(self.bn1_0(self.conv1_0(fea)))
            fea = self.relu(self.bn1_1(self.conv1_1(fea)))

            fea = self.relu(self.bn2_0(self.conv2_0(fea)))
            fea = self.relu(self.bn2_1(self.conv2_1(fea)))

            fea = self.relu(self.bn3_0(self.conv3_0(fea)))
            fea = self.relu(self.bn3_1(self.conv3_1(fea)))
        else:
            fea = self.relu(self.conv0_0(x))
            fea = self.relu(self.conv0_1(fea))

            fea = self.relu(self.conv1_0(fea))
            fea = self.relu(self.conv1_1(fea))

            fea = self.relu(self.conv2_0(fea))
            fea = self.relu(self.conv2_1(fea))

            fea = self.relu(self.conv3_0(fea))
            fea = self.relu(self.conv3_1(fea))

        fea = nn.AdaptiveAvgPool2d(1)(fea)
        fea = fea.view(fea.size(0), -1)

        out1 = self.classifier(fea)
        out2 = self.regressor(fea)

        return fea, out1, out2
