import matplotlib.pyplot as plt

import LearnTools
import Networks
from BatchMaker import *
# import argparse
import os
import time
import random
import torch.nn as nn
import math
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
# from matplotlib import pyplot as plt
import argparse
import wandb

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, phases_to_low = args.squash_phases, args.phases_low_res_idx
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor

if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.makedirs(ImageTools.progress_dir + progress_dir)

PATH_G = 'progress/' + progress_dir + '/g_weights.pth'
PATH_D = 'progress/' + progress_dir + '/d_weights.pth'
eta_file = 'eta.npy'

# G and D slices to choose from
g_batch_slices = [0]  # in 3D different views of the cube, better to keep it as
# 0..
d_batch_slices = [0]  # if D image is 3D, this has no impact, if it is a
# stack of 2D images (phasesXnum_imagesXwidthXhigth), then 0 should be chosen.

# adding 45 degree angle instead of z axis slices (TODO in addition)
forty_five_deg = False

# Root directory for dataset
dataroot = "data/"
D_image_path = 'train_cube_sofc.tif'
G_image_path = 'test_cube_sofc.tif'
D_image = dataroot + D_image_path
G_image = dataroot + G_image_path

# Number of workers for dataloader
workers = 2

# Batch sizes during training
if n_dims == 3:
    batch_size_G_for_D, batch_size_G, batch_size_D = 4, 32, 64
else:  # n_dims == 2
    batch_size_G_for_D, batch_size_G, batch_size_D = 64, 64, 64

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))

# the material indices to low-res:
to_low_idx = torch.LongTensor(phases_to_low).to(device)

# Number of channels in the training images. For color images this is 3
if squash:
    nc_g = 2
else:
    nc_g = 1 + to_low_idx.size()[0]  # channel for pore plus number of
    # material phases to low res.
nc_d = 3  # three phases for the discriminator input

# number of iterations in each epoch
epoch_iterations = 10000//batch_size_G

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Learning parameter for gradient penalty
Lambda = 10

# When to save progress
saving_num = 50


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_differences(network_g, high_res_im, save_dir, filename,
                     scale_factor, masks, with_deg=False):
    """
    Saves the image of the differences between the high-res real and the
    generated images that are supposed to be similar.
    """
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       to_low_idx,
                                                       scale_factor, device,
                                                       n_dims, squash)
    g_output = network_g(low_res_input).detach().cpu()
    images = [high_res_im.detach().cpu(),
              low_res_input.detach().cpu(), g_output]
    if with_deg:
        slices_45 = LearnTools.forty_five_deg_slices(masks, g_output)
        images.append(slices_45.detach().cpu())
    ImageTools.plot_fake_difference(images, save_dir, filename, with_deg)



if __name__ == '__main__':

    # 1. Start a new run
    wandb.init(project='SuperRes', config=args, name=progress_dir,
               entity='tldr-group')

    # The batch makers for D and G:
    BM_D = BatchMaker(device, path=D_image, sf=scale_f, dims=n_dims)
    BM_G = BatchMaker(device, path=G_image, sf=scale_f, dims=n_dims)

    nc_d = len(BM_D.phases)

    # Create the generator
    netG = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                              BM_G.scale_factor).to(device)
    wandb.watch(netG, log='all')

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Create the Discriminator
    netD = Networks.discriminator(ngpu, wd, nc_d, n_dims).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netG.apply(weights_init)
    # netD.apply(weights_init)
    # masks for 45 degree angle
    masks_45 = LearnTools.forty_five_deg_masks(batch_size_G_for_D,
                                               nc_d, BM_D.high_l)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    def generate_fake_image(detach_output=True):
        """
        :param detach_output: to detach the tensor output from gradient memory.
        :return: the generated image from G
        """
        # Generate batch of g input
        g_slice = random.choice(g_batch_slices)
        before_down_sampling = BM_G.random_batch_for_fake(batch_size_G_for_D,
                                                          g_slice)
        # down sample:
        low_res_im = LearnTools.down_sample_for_g_input(
            before_down_sampling, to_low_idx, BM_G.scale_factor,
            device, n_dims, squash)

        # Generate fake image batch with G
        if detach_output:
            return low_res_im, netG(low_res_im).detach()
        else:
            return low_res_im, netG(low_res_im)

    def take_fake_slices(fake_image, perm_idx):
        """
        :return: batch of slices from the 3d image (if 2d image,
        just returns the image)
        """
        if n_dims == 3:
            perm = perms_3d[perm_idx]
            if perm_idx == 2 and forty_five_deg:  # take forty five deg slices
                return LearnTools.forty_five_deg_slices(masks_45, fake_image)
            # permute the fake output of G to make it into a batch
            # of images to feed D (each time different axis)
            fake_slices_for_D = fake_image.permute(0, perm[0], 1, *perm[1:])
            # the new batch size feeding D:
            batch_size_new = batch_size_G_for_D * BM_G.high_l
            # reshaping for the correct size of D's input
            return fake_slices_for_D.view(batch_size_new, nc_d,
                                             BM_G.high_l, BM_G.high_l)
        else:  # same 2d slices are fed into D
            return fake_image

    # Training Loop!
    steps = epoch_iterations
    print("Starting Training Loop...")
    start = time.time()

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        i = 0

        j = np.random.randint(steps)  # to see different slices
        for _ in range(steps):

            ############################
            # (1) Update D network:
            ###########################

            _, fake_for_d = generate_fake_image(detach_output=True)

            for k in range(math.comb(n_dims, 2)):
                # only look at slices from the right directions:
                if not LearnTools.to_slice(k, forty_five_deg,
                                           D_dimensions_to_check):
                    continue
                # Train with all-real batch
                netD.zero_grad()
                # Batch of real high res for D
                d_slice = random.choice(d_batch_slices)
                high_res = BM_D.random_batch_for_real(batch_size_D, d_slice)

                # Forward pass real batch through D
                output_real = netD(high_res).view(-1).mean()

                # obtain fake slices from the fake image
                fake_slices = take_fake_slices(fake_for_d, k)

                # Classify all fake batch with D
                output_fake = netD(fake_slices).view(-1).mean()

                min_batch = min(high_res.size()[0], fake_slices.size()[0])
                # Calculate gradient penalty
                gradient_penalty = LearnTools.calc_gradient_penalty(netD,
                                   high_res[:min_batch], fake_slices[:min_batch],
                                   batch_size_D, BM_D.high_l, device,
                                   Lambda, nc_d)

                # discriminator is trying to minimize:
                d_cost = output_fake - output_real + gradient_penalty
                # Calculate gradients for D in backward pass
                d_cost.backward()
                optimizerD.step()

                wass = abs(output_fake.item() - output_real.item())

            ############################
            # (2) Update G network:
            ###########################

            if (i % g_update) == 0:
                netG.zero_grad()
                # generate fake again to update G:
                low_res, fake_for_g = generate_fake_image(detach_output=False)
                # save the cost of g to add from each axis:
                g_cost = 0
                # go through each axis
                for k in range(math.comb(n_dims, 2)):
                    # only look at slices from the right directions:
                    if not LearnTools.to_slice(k, forty_five_deg,
                                               D_dimensions_to_check):
                        continue
                    fake_slices = take_fake_slices(fake_for_g, k)
                    # perform a forward pass of all-fake batch through D
                    fake_output = netD(fake_slices).view(-1)
                    # get the pixel-wise-distance loss
                    pix_loss = LearnTools.pixel_wise_distance(low_res,
                               fake_for_g, to_low_idx, BM_G.scale_factor,
                               device, n_dims, squash)
                    # Calculate G's loss based on this output
                    if pix_loss.item() > 0.003:
                        g_cost += -fake_output.mean() + pix_distance * pix_loss
                    else:
                        g_cost += -fake_output.mean()



                # Calculate gradients for G
                g_cost.backward()
                # Update G
                optimizerG.step()
                wandb.log({"pixel distance": pix_loss})
                wandb.log({"wass": wass})
                wandb.log({"real": output_real, "fake": output_fake})

            # Output training stats
            if i == j or i == 0:
                ImageTools.calc_and_save_eta(steps, time.time(), start, i,
                                             epoch, num_epochs, eta_file)

                with torch.no_grad():  # only for plotting
                    save_differences(netG, BM_G.random_batch_for_fake(
                                     batch_size_G_for_D, random.choice(
                                      g_batch_slices)).detach(),
                                     progress_dir, 'running slices',
                                     BM_G.scale_factor, masks_45)
            i += 1
            print(i, j)

        if (epoch % 3) == 0:
            torch.save(netG.state_dict(), PATH_G)
            torch.save(netD.state_dict(), PATH_D)
            # wandb.save(PATH_G)
            # wandb.save(PATH_D)

    print('finished training')
