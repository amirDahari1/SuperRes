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

PATH_G = './g_test.pth'
PATH_D = './d_test.pth'
eta_file = 'eta.npy'

# Root directory for dataset
print(os.getcwd())
dataroot = "data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of channels in the training images. For color images this is 3
nc_g = 2  # two phases for the generator input
nc_d = 3  # three phases for the discriminator input

# Width generator channel hyperparameter
wd = 7
wg = 7

# Number of training epochs
num_epochs = 400

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
                                           shuffle=False, num_workers=workers)

g_dataloader = torch.utils.data.DataLoader(g_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=workers)
# TODO see maybe change to shuffle=true and normalize the data to have 0
#  mean and 1 std.

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))


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
        self.conv_minus_1 = nn.Conv2d(nc_d, 2 ** wg, 3, 1, 1)
        # first convolution, making many channels
        self.conv0 = nn.Conv2d(2 ** wg, 2 ** wg, 3, 1, 1)
        # the number of channels is because of pixel shuffling
        self.conv1 = nn.Conv2d(2 ** (wg - 2), 2 ** (wg - 2), 3, 1, 1)
        # last convolution, squashing all of the channels to 3 phases:
        self.conv2 = nn.Conv2d(2 ** (wg - 4), nc_d, 3, 1, 1)
        # use twice pixel shuffling:
        self.pixel_shuffling = torch.nn.PixelShuffle(2)
        # up samples
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear',
                               align_corners=False)

    def forward(self, x):

        # x after the first block:
        # TODO also instead of conv with 1 make it conv with 0
        x_first = nn.ReLU()(self.conv_minus_1(x))
        # making two more convolutions to understand the big areas:
        x_before1 = nn.ReLU()(self.conv0(x_first))
        # then after third time pixel shuffeling:
        x_block_0 = nn.ReLU()(self.pixel_shuffling(self.conv0(x_before1)))
        # x after two blocks:
        x_block_1 = nn.ReLU()(self.pixel_shuffling(self.conv1(x_block_0)))
        # upsampling of x and x_block_0:
        x_up = self.up1(x)
        # the concatenation of x, x_block_0 and x_block_1
        y = self.conv2(x_block_1)
        last_zero_size = list(x_up.size())
        last_zero_size[1] = 1
        last_zero_channel = torch.zeros(last_zero_size, dtype=x_up.dtype,
                                        device=x_up.device)
        x_up = torch.cat((x_up, last_zero_channel), dim=1)
        # y = torch.add(y, x_up)
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


def save_differences(network_g, high_res_im, rand_similarity, grey_idx,
                     device):
    """
    Saves the image of the differences between the high-res real and the
    generated images that are supposed to be similar.
    """
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       grey_idx, device)
    g_input = torch.cat((low_res_input, rand_similarity), dim=1)
    g_output = network_g(g_input).detach().cpu()
    ImageTools.plot_fake_difference(high_res_im.detach().cpu(),
                                    g_input.detach().cpu(), g_output)


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

    print("Starting Training Loop...")
    start = time.time()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        i = 0
        j = np.random.randint(saving_num//2)  # to see different slices
        steps = len(d_dataloader)
        for d_data, g_data in zip(d_dataloader, g_dataloader):

            ############################
            # (1) Update D network:
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            high_res = d_data[0].to(device)

            # ImageTools.show_gray_image(ImageTools.one_hot_decoding(
            #     high_res.detach().cpu())[0, :, :])
            # ImageTools.show_gray_image(high_res[0, 1, :, :].detach().cpu()*128)
            # Forward pass real batch through D
            output_real = netD(high_res).view(-1).mean()

            # Generate batch of latent vectors
            low_res = g_data[0].to(device)

            # create a random similarity channel
            init_rand = torch.rand(low_res.size()[0], 1, 1, 1).to(device)
            rand_sim = init_rand.repeat(1, 1, LOW_RES, LOW_RES)

            # concatenate the low-res image and the similarity scalar matrix
            low_res_with_sim = torch.cat((low_res, rand_sim), dim=1)

            # Generate fake image batch with G
            fake = netG(low_res_with_sim)

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
            pix_loss = LearnTools.pixel_wise_distance(low_res_with_sim,
                                                      fake, grey_index)

            # Calculate G's loss based on this output
            # g_cost = -fake_output.mean() + wass*pix_loss
            g_cost = -fake_output.mean() + 10 * pix_loss
            pixel_outputs.append(pix_loss.item())
            # Calculate gradients for G
            g_cost.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if (i + j) % saving_num == 0:
                torch.save(netG.state_dict(), PATH_G)
                torch.save(netG.state_dict(), PATH_D)
                ImageTools.graph_plot([real_outputs, fake_outputs],
                                ['real', 'fake'], '', 'LossesGraph')
                ImageTools.graph_plot([wass_outputs],
                                      ['wass'], '', 'WassGraph')
                ImageTools.graph_plot([pixel_outputs],
                                      ['pixel'], '', 'PixelLoss')
                ImageTools.graph_plot([gp_outputs], ['Gradient Penalty'], '',
                                      'GpGraph')
                ImageTools.calc_and_save_eta(steps, time.time(), start, i,
                                             epoch, num_epochs, eta_file)
                with torch.no_grad():  # only for plotting
                    save_differences(netG, high_res.detach(), rand_sim,
                                     grey_index, device)

            iters += 1
            i += 1
            print(i)

    # save the trained model

    print('finished training')
