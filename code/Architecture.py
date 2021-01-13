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

PATH_G = './g_test.pth'
PATH_D = './d_test.pth'

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
wd = 8
wg = 8

# Number of training epochs
num_epochs = 15

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# Learning parameter for gradient penalty
Lambda = 10

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
        self.conv_minus_1 = nn.Conv2d(nc_g, 2 ** wg, 3, 1, 1)
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
        # TODO maybe convolution twice on the 32x32 images to catch bigger
        #  9x9 areas of the image..
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
        # TODO addition instead of concatenation maybe?
        y = self.conv2(x_block_1)
        last_zero_size = list(x_up.size())
        last_zero_size[1] = 1
        last_zero_channel = torch.zeros(last_zero_size, dtype=x_up.dtype,
                                        device=x_up.device)
        x_up = torch.cat((x_up, last_zero_channel), dim=1)
        y = torch.add(y, x_up)
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
        # x = nn.ReLU()(self.break_conv1(x))

        # x = x.view(-1, 16*16)
        # return nn.Sigmoid()(self.linear(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        return self.conv5(x)


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

    # Print the model
    # print(netD)

    # print(netD.forward(test_run))

    # Initialize BCELoss function
    # criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    # real_label = 1.
    # fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    real_outputs = []  # the results of D on real images
    fake_outputs = []  # the results of D on fake images
    gp_outputs = []  # the gradient penalty outputs

    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        i = 0
        for d_data, g_data in zip(d_dataloader, g_dataloader):

            ############################
            # (1) Update D network:
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            high_res = d_data[0].to(device)
            # high_res.type(torch.FloatTensor)

            # Forward pass real batch through D
            output_real = netD(high_res).view(-1).mean()

            # Generate batch of latent vectors
            low_res = g_data[0].to(device)
            # Generate fake image batch with G
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
            gp_outputs.append(gradient_penalty.item())
            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()

            # Since we just updated D, perform another forward pass of
            # all-fake batch through D
            fake_output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            g_cost = -fake_output.mean()
            # Calculate gradients for G
            g_cost.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                ImageTools.graph_plot([real_outputs, fake_outputs],
                                ['real', 'fake'], '', 'LossGraph')
                ImageTools.graph_plot([gp_outputs], ['Gradient Penalty'], '',
                                      'GpGraph')

                # print(
                #     '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #     % (epoch, num_epochs, i, len(d_dataloader),
                #        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            print(i)
            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 100 == 0) or (
            #         (epoch == num_epochs - 1) and (i == len(d_dataloader) -
            #                                        1)):
            #     with torch.no_grad():
            #         fake = netG(low_res).detach().cpu()
            #         fake = ImageTools.fractions_to_ohe(fake)
            #         fake = ImageTools.one_hot_decoding(fake)
            #         ImageTools.show_gray_image(fake[0,:,:])
            #         ImageTools.show_gray_image(fake[1, :, :])

            iters += 1
            i += 1

    # save the trained model
    torch.save(netG.state_dict(), PATH_G)
    torch.save(netG.state_dict(), PATH_D)
    print('finished training')

    # # save the trained model
    # PATH = './g_test.pth'
    # torch.save(netG.state_dict(), PATH)

    # netG = Generator(ngpu)
    # netG.load_state_dict(torch.load(PATH))
    # high_res = next(iter(d_dataloader))[0]
    # print(high_res.shape)
    # high_res = ImageTools.one_hot_decoding(high_res)
    # print(high_res.shape)
    # low_res = ImageTools.cbd_to_grey(high_res)
    # low_res = ImageTools.down_sample(low_res)
    # low_res = np.expand_dims(low_res, axis=1)
    # print(low_res.shape)
    # input_to_g = ImageTools.one_hot_encoding(low_res)
    # print(low_res.shape)
    # fake = netG(torch.FloatTensor(input_to_g)).detach().cpu()
    # fake = ImageTools.fractions_to_ohe(fake)
    # fake = ImageTools.one_hot_decoding(fake)
    # ImageTools.show_three_by_two_gray(high_res, low_res.squeeze(), fake,
    #                                   'Very vanilla '
    #                                                               'super-res '
    #                                                            'results')
