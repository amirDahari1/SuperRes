import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, ngpu, wg, nc_g, nc_d, n_res_block, dims):
        super(Generator, self).__init__()
        if dims == 3:
            self.generator = Generator3D(ngpu, wg, nc_g, nc_d, n_res_block)
        else:  # dims == 2
            self.generator = Generator2D(ngpu, wg, nc_g, nc_d, n_res_block)

    def forward(self, x):
        return self.generator(x)

    def return_scale_factor(self, high_res_length):
        return self.generator.return_scale_factor(high_res_length)


# Generator Code
class Generator3D(nn.Module):
    def __init__(self, ngpu, wg, nc_g, nc_d, n_res_blocks):
        super(Generator3D, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.ngpu = ngpu
        self.conv_minus_1 = nn.Conv3d(nc_g, 2 ** wg, 3, 1, 1)
        self.bn_minus_1 = nn.BatchNorm3d(2**wg)
        # first convolution, making many channels
        self.conv_res = nn.ModuleList([nn.Conv3d(2 ** wg, 2 ** wg, 3, 1, 1)
                                       for _ in range(self.n_res_blocks)])
        self.bn_res = nn.ModuleList([nn.BatchNorm3d(2 ** wg)
                                     for _ in range(self.n_res_blocks)])
        # the number of channels is because of pixel shuffling
        self.conv_trans_1 = nn.ConvTranspose3d(2 ** wg, 2 ** (wg - 1), 4, 2, 2)
        self.bn1 = nn.BatchNorm3d(2 ** (wg - 1))
        # last convolution, squashing all of the channels to 3 phases:
        self.conv_trans_2 = nn.ConvTranspose3d(2 ** (wg - 1), nc_d,
                                               4, 2, 3)


    @staticmethod
    def res_block(x, conv, bn):
        """
        A forward pass of a residual block (from the original paper)
        :param x: the input
        :param conv: the convolution function, should return the same number
        of channels, and the same width and height of the image. For example,
        kernel size 3, padding 1 stride 1.
        :param bn: batch norm function
        :return: the result after the residual block.
        """
        # the residual side
        x_side = bn(conv(nn.ReLU()(bn(conv(x)))))
        return nn.ReLU()(x + x_side)

    @staticmethod
    def up_sample(x, conv, bn):
        """
        Up sampling with pixel shuffling block.
        """
        return nn.ReLU()(bn(conv(x)))

    def forward(self, x):
        """
        forward pass of x
        :param x: input
        :return: the output of the forward pass.
        """
        # x after the first run for many channels:
        x_first = nn.ReLU()(self.bn_minus_1(self.conv_minus_1(x)))
        # first residual block:
        after_block = self.res_block(x_first, self.conv_res[0], self.bn_res[0])
        # more residual blocks:
        for i in range(1, self.n_res_blocks):
            after_block = self.res_block(after_block, self.conv_res[i],
                                         self.bn_res[i])
        # skip connection to the end after all the blocks:
        after_res = x_first + after_block
        # up sampling with pixel shuffling (0):
        up_0 = self.up_sample(after_res, self.conv_trans_1, self.bn1)
        # up sampling with pixel shuffling (1):
        up_1 = self.conv_trans_2(up_0)
        return nn.Softmax(dim=1)(up_1)

    def return_scale_factor(self, high_res_length):
        return (high_res_length / 4 + 2) / high_res_length

class Discriminator(nn.Module):
    def __init__(self, ngpu, wd, nc_d, dims):
        super(Discriminator, self).__init__()
        if dims == 3:
            self.discriminator = Discriminator3d(ngpu, wd, nc_d)
        else:  # dims == 2
            self.discriminator = Discriminator2d(ngpu, wd, nc_d)

    def forward(self, x):
        return self.discriminator(x)


# Discriminator code
class Discriminator3d(nn.Module):
    def __init__(self, ngpu, wd, nc_d):
        super(Discriminator3d, self).__init__()
        self.ngpu = ngpu
        # first convolution, input is 3x66x66
        self.conv0 = nn.Conv2d(nc_d, 2 ** (wd - 3), 4, 2, 1)
        # second convolution, input is 32x32x32
        self.conv2 = nn.Conv2d(2 ** (wd - 3), 2 ** (wd - 2), 4, 2, 1)
        # third convolution, input is 64x16x16
        self.conv3 = nn.Conv2d(2 ** (wd - 2), 2 ** (wd - 1), 4, 2, 1)
        # fourth convolution, input is 128x8x8
        self.conv4 = nn.Conv2d(2 ** (wd - 1), 2 ** wd, 4, 2, 1)
        # fifth convolution, input is 256x4x4
        self.conv5 = nn.Conv2d(2 ** wd, 1, 4, 2, 0)

    def forward(self, x):
        x = nn.ReLU()(self.conv0(x))
        # x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        return self.conv5(x)


# Discriminator code
class Discriminator2d(nn.Module):
    def __init__(self, ngpu, wd, nc_d):
        super(Discriminator2d, self).__init__()
        self.ngpu = ngpu
        # first convolution, input is 3x128x128
        self.conv0 = nn.Conv2d(nc_d, 2 ** (wd - 4), 4, 2, 1)
        # first convolution, input is 4x64x64
        self.conv1 = nn.Conv2d(2 ** (wd - 4), 2 ** (wd - 3), 4, 2, 1)
        # second convolution, input is 8x32x32
        self.conv2 = nn.Conv2d(2 ** (wd - 3), 2 ** (wd - 2), 4, 2, 1)
        # third convolution, input is 32x16x16
        self.conv3 = nn.Conv2d(2 ** (wd - 2), 2 ** (wd - 1), 4, 2, 1)
        # fourth convolution, input is 64x8x8
        self.conv4 = nn.Conv2d(2 ** (wd - 1), 2 ** wd, 4, 2, 1)
        # fifth convolution, input is 128x4x4
        self.conv5 = nn.Conv2d(2 ** wd, 1, 4, 2, 0)

    def forward(self, x):
        x = nn.ReLU()(self.conv0(x))
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        return self.conv5(x)


# Generator Code
class Generator2D(nn.Module):
    def __init__(self, ngpu, wg, nc_g, nc_d, n_res_blocks):
        super(Generator2D, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.ngpu = ngpu
        self.conv_minus_1 = nn.Conv2d(nc_g, 2 ** wg, 3, 1, 1)
        self.bn_minus_1 = nn.BatchNorm2d(2**wg)
        # first convolution, making many channels
        self.conv_res = nn.ModuleList([nn.Conv2d(2 ** wg, 2 ** wg, 3, 1, 1)
                                       for _ in range(self.n_res_blocks)])
        self.bn_res = nn.ModuleList([nn.BatchNorm2d(2 ** wg) for _ in
                                     range(self.n_res_blocks)])
        # the number of channels is because of pixel shuffling
        self.conv1 = nn.Conv2d(2 ** (wg - 2), 2 ** (wg - 2), 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(2 ** (wg - 2))
        # last convolution, squashing all of the channels to 3 phases:
        self.conv2 = nn.Conv2d(2 ** (wg - 4), nc_d, 3, 1, 1)
        # use twice pixel shuffling:
        self.pixel = torch.nn.PixelShuffle(2)

    @staticmethod
    def res_block(x, conv, bn):
        """
        A forward pass of a residual block (from the original paper)
        :param x: the input
        :param conv: the convolution function, should return the same number
        of channels, and the same width and height of the image. For example,
        kernel size 3, padding 1 stride 1.
        :param bn: batch norm function
        :return: the result after the residual block.
        """
        # the residual side
        x_side = bn(conv(nn.ReLU()(bn(conv(x)))))
        return nn.ReLU()(x + x_side)

    @staticmethod
    def up_sample(x, pix_shuffling, conv, bn):
        """
        Up sampling with pixel shuffling block.
        """
        return nn.ReLU()(pix_shuffling(bn(conv(x))))

    def forward(self, x):
        """
        forward pass of x
        :param x: input
        :return: the output of the forward pass.
        """
        # x after the first run for many channels:
        # TODO also instead of conv with 1 padding to conv with 0 padding
        x_first = nn.ReLU()(self.bn_minus_1(self.conv_minus_1(x)))
        # first residual block:
        after_block = self.res_block(x_first, self.conv_res[0], self.bn_res[0])
        # more residual blocks:
        for i in range(1, self.n_res_blocks):
            after_block = self.res_block(after_block, self.conv_res[i],
                                         self.bn_res[i])
        # skip connection to the end after all the blocks:
        after_res = x_first + after_block
        # up sampling with pixel shuffling (0):
        up_0 = self.up_sample(after_res, self.pixel, self.bn0, self.conv0)
        # up sampling with pixel shuffling (1):
        up_1 = self.up_sample(up_0, self.pixel, self.bn1, self.conv1)

        y = self.conv2(up_1)

        # TODO maybe different function in the end?
        return nn.Softmax(dim=1)(y)

    def return_scale_factor(self, high_res_length):
        return (high_res_length / 4) / high_res_length