import LearnTools
import Networks
from BatchMaker import *
import wandb
import argparse
import os
import time
import random
import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from matplotlib import pyplot as plt
import argparse

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update = args.num_epochs, args.g_update

# 1. Start a new run
# wandb.init(project='wandb test', config=args, name=progress_dir)

if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.makedirs(ImageTools.progress_dir + progress_dir)

PATH_G = 'progress/' + progress_dir + '/g_weights.pth'
PATH_D = 'progress/' + progress_dir + '/d_weights.pth'
eta_file = 'eta.npy'

# G and D slices to choose from
g_slices = [0, 1]
d_slices = [0, 1]

# Root directory for dataset
dataroot = "data/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of channels in the training images. For color images this is 3
nc_g = 2  # two phases for the generator input
nc_d = 3  # three phases for the discriminator input

# number of iterations in each epoch
epoch_iterations = 10000//batch_size

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Learning parameter for gradient penalty
Lambda = 10

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# When to save progress
saving_num = 50

# Create the datasets for the training of d and g
# d_train_dataset = torch.load(dataroot + 'd_train.pth')
# g_train_dataset = torch.load(dataroot + 'g_train.pth')

# Create the dataloader
# d_dataloader = torch.utils.data.DataLoader(d_train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True, num_workers=workers)

# g_dataloader = torch.utils.data.DataLoader(g_train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True, num_workers=workers)
# TODO see maybe change to shuffle=true and normalize the data to have 0
#  mean and 1 std.

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))

# the grey channel in the images:
grey_index = torch.LongTensor([1]).to(device)

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


def save_differences(network_g, high_res_im, grey_idx,
                     device, save_dir, filename, wandb):
    """
    Saves the image of the differences between the high-res real and the
    generated images that are supposed to be similar.
    """
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       grey_idx, device)
    g_output = network_g(low_res_input).detach().cpu()
    ImageTools.plot_fake_difference(high_res_im.detach().cpu(),
                                    low_res_input.detach().cpu(), g_output,
                                    save_dir, filename, wandb)


def save_tif_3d(network_g, high_res_im, grey_idx, device, filename):
    """
        Saves a tif image of the output of G on all of the 3d image high_res_im
    """
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       grey_idx, device)
    g_output = network_g(low_res_input).detach().cpu()
    g_output_grey = ImageTools.one_hot_decoding(g_output).astype('uint8')
    imsave('progress/' + progress_dir + '/' + filename, g_output_grey)
    low_res_grey = ImageTools.one_hot_decoding(low_res_input).astype('uint8')
    imsave('progress/' + progress_dir + '/low_res' + filename , low_res_grey)
    high_res_im = ImageTools.one_hot_decoding(high_res_im).astype('uint8')
    imsave('progress/' + progress_dir + '/' + filename + '-original',
           high_res_im)


if __name__ == '__main__':
    # The batch maker:
    BM = BatchMaker(device)

    # Create the generator
    netG = Networks.Generator2D(ngpu, wg, nc_g, nc_d, n_res_blocks).to(device)
    # wandb.watch(netG)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Networks.Discriminator(ngpu, wd, nc_d).to(device)

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

    steps = epoch_iterations

    print("Starting Training Loop...")
    start = time.time()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        i = 0

        j = np.random.randint(steps)  # to see different slices
        for _ in range(steps):

            ############################
            # (1) Update D network:
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            # high_res = d_data[0].to(device)
            d_slice = random.choice(d_slices)
            high_res = BM.random_batch(batch_size, d_slice)

            # Forward pass real batch through D
            output_real = netD(high_res).view(-1).mean()

            # Generate batch of g input
            g_slice = random.choice(g_slices)
            before_down_sampling = BM.random_batch(batch_size, g_slice)
            # down sample:
            low_res = LearnTools.down_sample_for_g_input(
                before_down_sampling, grey_index, device)

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

            if (i % g_update) == 0:
                netG.zero_grad()

                # Since we just updated D, perform another forward pass of
                # all-fake batch through D
                fake_output = netD(fake).view(-1)
                # get the pixel-wise-distance loss
                pix_loss = LearnTools.pixel_wise_distance(low_res,
                                                          fake, grey_index)

                # Calculate G's loss based on this output
                # g_cost = -fake_output.mean()
                g_cost = -fake_output.mean() + pix_distance * pix_loss
                pixel_outputs.append(pix_loss.item())
                # Calculate gradients for G
                g_cost.backward()
                # Update G
                optimizerG.step()
            else:  # not a g update iteration
                pix_loss = LearnTools.pixel_wise_distance(low_res,
                                                          fake, grey_index)
                pixel_outputs.append(pix_loss.item())

            # Output training stats
            if i == j:
                wandb.log({"wass": wass})
                wandb.log({"real": output_real, "fake": output_fake})
                wandb.log({"pixel distance": pix_loss})
                torch.save(netG.state_dict(), PATH_G)
                torch.save(netD.state_dict(), PATH_D)
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
                                     'running slices', wandb)
                # save fifteen images during the run
                if epoch % (num_epochs//21) == 0 and epoch > 0:
                    save_differences(netG, high_res.detach(), grey_index,
                                     device, progress_dir, 'Iteration_'
                                     + str(iters), wandb)

            iters += 1
            i += 1
            # print(i)

    print('finished training')
