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

#################################################################
# All variables initialization, for the training loop skip ahead.
#################################################################

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, phases_to_low = args.squash_phases, args.phases_low_res_idx
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
rotation, anisotropic = args.with_rotation, args.anisotropic
rotations_bool, down_sample = args.rotations_bool, args.down_sample
separator = args.separator

if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.makedirs(ImageTools.progress_dir + progress_dir)

PATH_G = 'progress/' + progress_dir + '/g_weights.pth'
PATH_D = 'progress/' + progress_dir + '/d_weights.pth'
eta_file = 'eta.npy'

# Root directory for dataset
dataroot = "data/"

D_images = [dataroot + d_path for d_path in args.d_image_path]
G_image = dataroot + args.g_image_path

# Number of workers for dataloader
workers = 2

# G and D slices to choose from
g_batch_slices = [0]  # in 3D different views of the cube, better to keep it as
# 0..

# adding 45 degree angle instead of z axis slices
forty_five_deg = False

# Batch sizes during training
if n_dims == 3:
    batch_size_G_for_D, batch_size_G, batch_size_D = 4, 32, 64
else:  # n_dims == 2
    batch_size_G_for_D, batch_size_G, batch_size_D = 64, 64, 64

# Number of GPUs available. Use 0 for CPU mode. For more than 1 GPU parallel
# computing, this feature needs to be updated.
ngpu = 1

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device is ' + str(device))

# the material indices to low-res:
to_low_idx = torch.LongTensor(phases_to_low).to(device)

# Number of channels in the training images.
if squash:
    nc_g = 2 + 1
else:
    nc_g = 1 + to_low_idx.size()[0] + 1  # channel for pore plus number of
    # material phases to low res plus noise channel.

# number of iterations in each epoch
epoch_iterations = 10000 // batch_size_G

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Learning parameter for gradient penalty
Lambda = 10

# When to save progress
saving_num = 50


def save_differences_and_metrics(input_to_g, output_of_g, save_dir, filename,
                                 masks, hr_metrics, with_deg=False):
    """
    Saves the image of the differences between the high-res real and the
    generated images that are supposed to be similar.
    """
    images = [input_to_g.clone().detach().cpu()]
    g_output = output_of_g.cpu()
    ImageTools.log_metrics(g_output, hr_metrics)
    images = images + [input_to_g.detach().cpu(), g_output]
    if with_deg:
        slices_45 = LearnTools.forty_five_deg_slices(masks, g_output)
        images.append(slices_45.detach().cpu())
    ImageTools.plot_fake_difference(images, save_dir, filename, with_deg)


if __name__ == '__main__':

    # 1. Start a new run
    wandb.init(project='SuperRes', config=args, name=progress_dir,
               entity='tldr-group')

    # The batch makers for D and G:
    D_BMs, D_nets, D_optimisers = Networks. \
        return_D_nets(ngpu, wd, n_dims, device, lr, beta1, anisotropic,
                      D_images, scale_f, rotation, rotations_bool)
    # Number of HR number of phases:
    nc_d = len(D_BMs[0].phases)
    # volume fraction and surface area high-res metrics:
    hr_metrics = D_BMs[0].hr_metrics

    BM_G = BatchMaker(device=device, to_low_idx=to_low_idx, path=G_image,
                      sf=scale_f, dims=n_dims, stack=False,
                      down_sample=down_sample, low_res=not down_sample,
                      rot_and_mir=False, squash=squash)

    # Create the generator
    netG = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                              BM_G.scale_factor).to(device)
    wandb.watch(netG, log='all')

    # Create the down-sample object to compare between super-res and low-res
    down_sample_object = LearnTools. \
        DownSample(squash, n_dims, to_low_idx, scale_f, device, separator).to(
        device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # masks for 45 degree angle
    masks_45 = LearnTools.forty_five_deg_masks(batch_size_G_for_D,
                                               nc_d, D_BMs[0].high_l)

    # Setup Adam optimizers for G
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    def generate_fake_image(detach_output=True):
        """
        :param detach_output: to detach the tensor output from gradient memory.
        :return: the generated image from G
        """
        # Generate batch of G's input:
        g_slice = random.choice(g_batch_slices)
        input_to_G = BM_G.random_batch_for_fake(batch_size_G_for_D, g_slice)
        input_size = input_to_G.size()
        # make noise channel and concatenate it to input:
        noise = torch.randn(input_size[0], 1, *input_size[2:], device=device)
        input_to_G = torch.cat((input_to_G, noise), dim=1)

        # Generate fake image batch with G
        if detach_output:
            return input_to_G, netG(input_to_G).detach()
        else:
            return input_to_G, netG(input_to_G)


    def take_fake_slices(fake_image, perm_idx):
        """
        :param fake_image: The fake image to slice at all directions.
        :param perm_idx: The permutation index for permutation before slicing.
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
            batch_size_new = batch_size_G_for_D * D_BMs[0].high_l
            # reshaping for the correct size of D's input
            return fake_slices_for_D.reshape(batch_size_new, nc_d,
                                             D_BMs[0].high_l, D_BMs[0].high_l)
        else:  # same 2d slices are fed into D
            return fake_image


    ################
    # Training Loop!
    ################

    steps = epoch_iterations
    print("Starting Training Loop...")
    start = time.time()

    for epoch in range(num_epochs):

        j = np.random.randint(steps)  # to see different slices
        for i in range(steps):

            #######################
            # (1) Update D network:
            #######################

            _, fake_for_d = generate_fake_image(detach_output=True)

            for k in range(math.comb(n_dims, 2)):
                BM_D, netD, optimizerD = D_BMs[k], D_nets[k], D_optimisers[k]
                # only look at slices from the right directions:
                if not LearnTools.to_slice(k, forty_five_deg,
                                           D_dimensions_to_check):
                    continue

                # Train with all-real batch
                netD.zero_grad()

                # Batch of real high res for D
                high_res = BM_D.random_batch_for_real(batch_size_D)

                # Forward pass real batch through D
                output_real = netD(high_res).view(-1).mean()

                # obtain fake slices from the fake image
                fake_slices = take_fake_slices(fake_for_d, k)

                # Classify all fake batch with D
                output_fake = netD(fake_slices).view(-1).mean()

                min_batch = min(high_res.size()[0], fake_slices.size()[0])
                # Calculate gradient penalty
                gradient_penalty = LearnTools.\
                    calc_gradient_penalty(netD, high_res[:min_batch],
                                          fake_slices[:min_batch],
                                          batch_size_D, BM_D.high_l, device,
                                          Lambda, nc_d)

                # discriminator is trying to minimize:
                d_cost = output_fake - output_real + gradient_penalty
                # Calculate gradients for D in backward pass
                d_cost.backward()
                optimizerD.step()

                wass = abs(output_fake.item() - output_real.item())

            #######################
            # (2) Update G network:
            #######################

            if (i % g_update) == 0:
                netG.zero_grad()
                # generate fake again to update G:
                low_res, fake_for_g = generate_fake_image(detach_output=False)
                # save the cost of g to add from each axis:
                g_cost = 0
                # go through each axis
                for k in range(math.comb(n_dims, 2)):
                    netD, optimizerD = D_nets[k], D_optimisers[k]
                    # only look at slices from the right directions:
                    if not LearnTools.to_slice(k, forty_five_deg,
                                               D_dimensions_to_check):
                        continue
                    fake_slices = take_fake_slices(fake_for_g, k)
                    # perform a forward pass of all-fake batch through D
                    fake_output = netD(fake_slices).view(-1).mean()

                    if k == 0:
                        wandb.log({'yz_slice': fake_output})
                    if k == 1:
                        wandb.log({'xz_slice': fake_output})
                    if k == 2 and forty_five_deg:
                        wandb.log({'deg_slice': fake_output})
                        fake_output = fake_output * 100

                    # get the voxel-wise-distance loss
                    low_res_without_noise = low_res[:, :-1]  # without noise
                    pix_loss = down_sample_object.voxel_wise_distance(
                        fake_for_g, low_res_without_noise)

                    # Calculate G's loss based on this output
                    if pix_loss.item() > 0.005:
                        g_cost += -fake_output + pix_distance * pix_loss
                    else:
                        g_cost += -fake_output

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
                    g_input_plot, g_output_plot = generate_fake_image(
                        detach_output=True)
                    # plot input without the noise channel
                    save_differences_and_metrics\
                        (g_input_plot[:, :-1], g_output_plot, progress_dir,
                         'running slices', masks_45, hr_metrics,
                         forty_five_deg)
            print(i, j)

        if (epoch % 3) == 0:
            torch.save(netG.state_dict(), PATH_G)
            wandb.save(PATH_G)

    print('finished training')
