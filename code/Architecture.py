from BatchMaker import *
import LearnTools

import argparse
import os
import time
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
import argparse

# Parsing arguments:
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='default',
                    help='Stores the progress output in the \
                    directory name given')
parser.add_argument('-wd', '--widthD', type=int, default=8,
                    help='Hyper-parameter for \
                    the width of the Discriminator network')
parser.add_argument('-wg', '--widthG', type=int, default=8,
                    help='Hyper-parameter for the \
                    width of the Generator network')
parser.add_argument('-n_res', '--n_res_blocks', type=int, default=3,
                    help='Number of residual blocks in the network.')
args = parser.parse_args()

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks = args.n_res_blocks

if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.mkdir(ImageTools.progress_dir + progress_dir)

PATH_G = './g_test.pth'
PATH_D = './d_test.pth'
eta_file = 'eta.npy'

# pixel loss average
pix_loss_average = 0.0434

# Root directory for dataset
dataroot = "data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of channels in the training images. For color images this is 3
nc_g = 2  # two phases for the generator input
nc_d = 3  # three phases for the discriminator input

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# Learning parameter for gradient penalty
Lambda = 10

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# When to save progress
saving_num = 50

# Create the datasets for the training of d and g
d_train_dataset = torch.load(dataroot + 'd_train.pth')
g_train_dataset = torch.load(dataroot + 'g_train.pth')

# Create the dataloader
d_dataloader = torch.utils.data.DataLoader(d_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=workers)

g_dataloader = torch.utils.data.DataLoader(g_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=workers)
# TODO see maybe change to shuffle=true and normalize the data to have 0
#  mean and 1 std.

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))


# Plot one training image of d
# first_d_batch = next(iter(d_dataloader))

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
        self.conv_minus_1 = nn.Conv2d(nc_g, 2 ** wg, 3, 1, 1)
        self.bn_minus_1 = nn.BatchNorm2d(2**wg)
        # first convolution, making many channels
        self.conv0 = nn.Conv2d(2 ** wg, 2 ** wg, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(2 ** wg)
        # the number of channels is because of pixel shuffling
        self.conv1 = nn.Conv2d(2 ** (wg - 2), 2 ** (wg - 2), 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(2 ** (wg - 2))
        # last convolution, squashing all of the channels to 3 phases:
        self.conv2 = nn.Conv2d(2 ** (wg - 4), nc_d, 3, 1, 1)
        # use twice pixel shuffling:
        self.pixel = torch.nn.PixelShuffle(2)
        # up samples
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear',
                               align_corners=False)

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
        after_block = self.res_block(x_first, self.conv0, self.bn0)
        # more residual blocks:
        for i in range(n_res_blocks - 1):
            after_block = self.res_block(after_block, self.conv0, self.bn0)
        # skip connection to the end after all the blocks:
        after_res = x_first + after_block
        # up sampling with pixel shuffling (0):
        up_0 = self.up_sample(after_res, self.pixel, self.bn0, self.conv0)
        # up sampling with pixel shuffling (1):
        up_1 = self.up_sample(up_0, self.pixel, self.bn1, self.conv1)

        y = self.conv2(up_1)
        
        # TODO maybe different function in the end?
        return nn.Softmax(dim=1)(y)


# Discriminator code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
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


def save_differences(network_g, high_res_im, grey_idx,
                     device, save_dir, filename):
    """
    Saves the image of the differences between the high-res real and the
    generated images that are supposed to be similar.
    """
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       grey_idx, device)
    # g_input = torch.cat((low_res_input, rand_similarity), dim=1)
    g_output = network_g(low_res_input).detach().cpu()
    ImageTools.plot_fake_difference(high_res_im.detach().cpu(),
                                    low_res_input.detach().cpu(), g_output,
                                    save_dir, filename)


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
    # print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)


    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    wass_outputs = []
    real_outputs = []  # the results of D on real images
    fake_outputs = []  # the results of D on fake images
    gp_outputs = []  # the gradient penalty outputs
    pixel_outputs = []

    iters = 0
    # the grey channel in the images:
    grey_index = torch.LongTensor([1]).to(device)
    steps = len(d_dataloader)

    print("Starting Training Loop...")
    start = time.time()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        i = 0

        j = np.random.randint(steps)  # to see different slices
        for d_data, g_data in zip(d_dataloader, g_dataloader):

            ############################
            # (1) Update D network:
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            high_res = d_data[0].to(device)

            # Forward pass real batch through D
            output_real = netD(high_res).view(-1).mean()

            # Generate batch of latent vectors
            low_res = g_data[0].to(device)

            # create a random similarity channel
            # init_rand = torch.rand(low_res.size()[0], 1, 1, 1).to(device)
            # rand_sim = init_rand.repeat(1, 1, LOW_RES, LOW_RES)

            # concatenate the low-res image and the similarity scalar matrix
            # low_res_with_sim = torch.cat((low_res, rand_sim), dim=1)

            # Generate fake image batch with G
            # fake = netG(low_res_with_sim)
            fake = netG(low_res)

            # Classify all fake batch with D
            output_fake = netD(fake.detach()).view(-1).mean()

            # Calculate gradient penalty
            gradient_penalty = LearnTools.calc_gradient_penalty(netD,
                               high_res, fake.detach(), batch_size, HIGH_RES,
                               device, Lambda, nc_d)

            # discriminator is trying to minimize:
            d_cost = output_fake - output_real + gradient_penalty
            # Calculate gradients for D in backward pass
            d_cost.backward()
            optimizerD.step()

            # save the outputs

            real_outputs.append(output_real.item())
            fake_outputs.append(output_fake.item())
            wass = abs(output_fake.item() - output_real.item())
            wass_outputs.append(wass)
            gp_outputs.append(gradient_penalty.item())
            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()

            # Since we just updated D, perform another forward pass of
            # all-fake batch through D
            fake_output = netD(fake).view(-1)
            # get the pixel-wise-distance loss
            pix_loss = LearnTools.pixel_wise_distance(low_res,
                                                      fake, grey_index)

            # Calculate G's loss based on this output
            # g_cost = -fake_output.mean()
            g_cost = -fake_output.mean() + 10 * pix_loss
            pixel_outputs.append(pix_loss.item())
            print(pixel_outputs)
            # Calculate gradients for G
            g_cost.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if i == j:
                torch.save(netG.state_dict(), PATH_G)
                torch.save(netG.state_dict(), PATH_D)
                ImageTools.graph_plot([real_outputs, fake_outputs],
                                      ['real', 'fake'], progress_dir,
                                      'LossesGraphBN')
                ImageTools.graph_plot([wass_outputs],
                                      ['wass'], progress_dir, 'WassGraphBN')
                ImageTools.graph_plot([pixel_outputs],
                                      ['pixel'], progress_dir, 'PixelLossBN')
                ImageTools.graph_plot([gp_outputs], ['Gradient Penalty'],
                                      progress_dir, 'GpGraphBN')
                ImageTools.calc_and_save_eta(steps, time.time(), start, i,
                                             epoch, num_epochs, eta_file)
                with torch.no_grad():  # only for plotting
                    save_differences(netG, high_res.detach(),
                                     grey_index, device, progress_dir,
                                     'running slices')
                # save fifteen images during the run
                if epoch % (num_epochs//21) == 0 and epoch > 0:
                    save_differences(netG, high_res.detach(), grey_index,
                                     device, progress_dir, 'Iteration_'
                                     + str(iters))

            iters += 1
            i += 1
            print(i)

    # save the trained model

    print('finished training')
