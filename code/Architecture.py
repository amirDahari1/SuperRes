from BatchMaker import *

import argparse
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt


# Root directory for dataset
dataroot = "data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of channels in the training images. For color images this is 3
nc_g = 2  # two phases for the generator input
nc_d = 3  # three phases for the discriminator input

# Width generator channel hyperparameter
wg = 7

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Create the datasets for the training of d and g
d_train_dataset = torch.load(dataroot + 'd_train.pth')
g_train_dataset = torch.load(dataroot + 'g_train.pth')

# Create the dataloader
d_dataloader = torch.utils.data.DataLoader(d_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=workers)

g_dataloader = torch.utils.data.DataLoader(g_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=workers)
# TODO see maybe change to shuffle=true and normalize the data to have 0
#  mean and 1 std.

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot one training image of d
# first_d_batch = next(iter(d_dataloader))

# np_d_decode = BatchMaker.one_hot_decoding(first_d_batch[0])
# print(np_d_decode.shape)
# BatchMaker.show_image(np_d_decode[10, :, :])


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # first convolution, making many channels
        self.conv0 = nn.Conv2d(nc_g, 2**wg, 3, 1, 1)
        # the number of channels is because of pixel shuffling
        self.conv1 = nn.Conv2d(2**(wg-2), 2**(wg-2), 3, 1, 1)
        # last convolution, squashing all of the channels to 3 phases:
        self.conv2 = nn.Conv2d(2+2**(wg-2)+2**(wg-4), nc_d, 3, 1, 1)
        # use twice pixel shuffling:
        self.pixel_shuffling = torch.nn.PixelShuffle(2)
        # up samples
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear',
                               align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)

    def forward(self, x):
        # TODO maybe convolution twice on the 32x32 images to catch bigger
        #  9x9 areas of the image..
        # x after the first block:
        x_block_0 = nn.PReLU()(self.pixel_shuffling(self.conv0(x)))
        # x after two blocks:
        x_block_1 = nn.PReLU()(self.pixel_shuffling(self.conv1(x_block_0)))
        # upsampling of x and x_block_0:
        x_up = self.up1(x)
        x_block_0_up = self.up2(x_block_0)
        # the concatenation of x, x_block_0 and x_block_1
        # TODO addition instead of concatenation maybe?
        y = torch.cat((x_up, x_block_0_up, x_block_1), dim=1)
        # TODO maybe different function in the end?
        return nn.Softmax(dim=1)(self.conv2(y))

# Discriminator code


if __name__ == '__main__':
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # see if the dimensions are correct
    first_g_batch = next(iter(g_dataloader))
    before_run = ImageTools.one_hot_decoding(first_g_batch[0])

    test_run = netG.forward(first_g_batch[0]).detach()
    print(test_run.shape)
    test_run_gray = ImageTools.one_hot_decoding(ImageTools.fractions_to_ohe(
        test_run))

    ImageTools.show_three_by_two_gray(before_run[0:3,:,:],test_run_gray[0:3,
                                                          :,:], 'Generator '
                                                                'run without training')
    # print an image


