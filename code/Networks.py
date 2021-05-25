import numpy as np
import torch.nn as nn
from torch.nn.functional import interpolate
import torch
import torch.optim as optim
import copy
import math
from BatchMaker import *
smaller_cube = False
EPS = 10e-10
modes = ['bilinear', 'trilinear']


def return_D_nets(ngpu, wd, nc_d, n_dims, device, lr, beta1, anisotropic,
                  D_images, scale_f, rotation):
    D_nets = []
    D_optimisers = []
    D_BMs = []
    if anisotropic:
        rotation = [False, False, True]  # TODO change this to be general!!
        for i in np.arange(n_dims):
            BM_D = BatchMaker(device, path=D_images[i], sf=scale_f,
                              dims=n_dims, stack=True, low_res=False,
                              rot_and_mir=rotation[i])
            # Create the Discriminator
            netD = discriminator(ngpu, wd, nc_d, n_dims).to(device)
            # Handle multi-gpu if desired
            if (device.type == 'cuda') and (ngpu > 1):
                netD = nn.DataParallel(netD, list(range(ngpu)))
            optimiserD = optim.Adam(netD.parameters(), lr=lr,
                                    betas=(beta1, 0.999))
            D_BMs.append(BM_D)
            D_nets.append(netD)
            D_optimisers.append(optimiserD)

    else:  # isotropic
        # Create the batch maker
        BM_D = BatchMaker(device, path=D_images[0], sf=scale_f,
                          dims=n_dims, stack=True, low_res=False,
                          rot_and_mir=rotation)
        # Create the Discriminator
        netD = discriminator(ngpu, wd, nc_d, n_dims).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))
        optimiserD = optim.Adam(netD.parameters(), lr=lr,
                                betas=(beta1, 0.999))
        D_BMs = [BM_D]*n_dims  # same batch maker, different pointers
        D_nets = [netD]*n_dims  # same network, different pointers
        D_optimisers = [optimiserD]*n_dims  # same optimiser, different
        # pointers
    return D_BMs, D_nets, D_optimisers


def generator(ngpu, wg, nc_g, nc_d, n_res_block, dims, scale_factor):
    if dims == 3:
        return Generator3D(ngpu, wg, nc_g, nc_d, n_res_block, scale_factor)
    else:  # dims == 2
        return Generator2D(ngpu, wg, nc_g, nc_d, n_res_block)


# Generator Code
class Generator3D(nn.Module):
    def __init__(self, ngpu, wg, nc_g, nc_d, n_res_blocks, scale_factor):
        super(Generator3D, self).__init__()
        self.scale_factor = scale_factor
        self.n_res_blocks = n_res_blocks
        self.ngpu = ngpu
        # how to change the channels depends on the number of layers
        sf_c = int(math.log(self.scale_factor - EPS, 2))
        wg_sf = wg + sf_c
        # first convolution, making many channels
        self.conv_minus_1 = nn.Conv3d(nc_g, 2 ** (wg_sf-1), 3, stride=1,
                                      padding=1, padding_mode='replicate')
        self.bn_minus_1 = nn.BatchNorm3d(2**(wg_sf - 1))

        self.conv_res = nn.ModuleList([nn.Conv3d(2 ** (wg_sf - 1), 2 ** (wg_sf - 1), 3,
                                                 stride=1, padding=1,
                                                 padding_mode='replicate')
                                       for _ in range(self.n_res_blocks*2)])
        self.bn_res = nn.ModuleList([nn.BatchNorm3d(2 ** (wg_sf - 1))
                                     for _ in range(self.n_res_blocks*2)])

        # convolution resize before t-conv (if scale-ratio > 4):
        if self.scale_factor > 4:
            self.conv_resize_0 = nn.Conv3d(2 ** (wg_sf - 1), 2 ** (wg_sf - 2), 3,
                                           stride=1,
                                           padding=1, padding_mode='replicate')
            self.bn_resize_0 = nn.BatchNorm3d(2 ** (wg_sf - 2))
            wg_sf -= 1
        # transpose convolution:
        if self.scale_factor > 2:
            self.conv_trans = nn.ConvTranspose3d(2 ** (wg_sf - 1),
                                             2 ** (wg_sf - 2), 4, 2, 1)
            self.bn_trans = nn.BatchNorm3d(2 ** (wg_sf - 2))
            wg_sf -= 1
        # convolution resize:
        self.conv_resize = nn.Conv3d(2 ** (wg_sf - 1), 2 ** (wg_sf - 2),
                                     3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.bn_resize = nn.BatchNorm3d(2 ** (wg_sf - 2))
        self.conv_bf_end = nn.Conv3d(2 ** (wg_sf - 2), nc_d, 3, stride=1,
                                     padding=1, padding_mode='replicate')
        # self.conv_concat = nn.Conv3d(nc_d+nc_g, nc_d, 1, 1, 0)

    @staticmethod
    def res_block(x, bn_out, conv_out, bn_in, conv_in):
        """
        A forward pass of a residual block (from the original paper)
        :return: the result after the residual block. the convolution
        function should return the same number of channels, and the same
        width and height of the image. For example, kernel size 3, padding 1
        stride 1.
        """
        # the residual side
        x_side = bn_out(conv_out(nn.ReLU()(bn_in(conv_in(x)))))
        return nn.ReLU()(x + x_side)

    def forward(self, x):
        """
        forward pass of x
        :param x: input
        :param mask: for plotting purposes, returns the mask (result of all
        convolutions) and the up-sampled original image.
        :return: the output of the forward pass.
        """
        cur_scale = copy.copy(self.scale_factor)  # current scale factor of
        # the image on the forward run.
        # x after the first run for many channels:
        x_first = nn.ReLU()(self.bn_minus_1(self.conv_minus_1(x)))
        # first residual block:
        after_block = self.res_block(x_first, self.bn_res[0], self.conv_res[0],
                                     self.bn_res[1], self.conv_res[1])
        # more residual blocks:
        for i in range(2, self.n_res_blocks, 2):
            after_block = self.res_block(after_block, self.bn_res[i],
                                         self.conv_res[i], self.bn_res[i+1],
                                         self.conv_res[i+1])
        # skip connection to the end after all the blocks:
        res = x_first + after_block
        # up sampling using conv resize
        if 4 < cur_scale <= 8:
            up_sample = nn.Upsample(scale_factor=2, mode=modes[1])  # TODO
            # maybe TODO trilinear upsampling?
            res = nn.ReLU()(self.bn_resize_0(self.conv_resize_0(up_sample(
                res))))
            cur_scale /= 2
        # up sampling using transpose convolution
        if 2 < cur_scale <= 4:
            res = nn.ReLU()(self.bn_trans(self.conv_trans(res)))
            cur_scale /= 2
        # last up sample using conv resize:
        up_sample = nn.Upsample(scale_factor=cur_scale, mode=modes[1])  # TODO
        # maybe TODO trilinear upsampling?
        super_res = nn.ReLU()(self.bn_resize(self.conv_resize(up_sample(res))))
        # another convolution before the end:
        bf_end = self.conv_bf_end(super_res)
        # softmax of the phase dimension:
        return nn.Softmax(dim=1)(bf_end)


def discriminator(ngpu, wd, nc_d, dims):
    if dims == 3:
        return Discriminator3d(ngpu, wd, nc_d)
    else:  # dims == 2
        return Discriminator2d(ngpu, wd, nc_d)


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
        # for smaller cube
        self.conv_early = nn.Conv2d(2**(wd-1), 1, 4, 2, 0)

    def forward(self, x):
        x = nn.ReLU()(self.conv0(x))
        # x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        if smaller_cube:
            return self.conv_early(x)
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