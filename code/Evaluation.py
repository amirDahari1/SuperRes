import BatchMaker
import LearnTools
import Networks
import ImageTools
import argparse
import torch
import numpy as np
from tifffile import imsave

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, phases_to_low = args.squash_phases, args.phases_low_res_idx
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor


# progress_main_dir = 'progress/' + progress_dir
progress_main_dir = 'progress'
# path_to_g_weights = progress_main_dir + '/g_weights.pth'
path_to_g_weights = progress_main_dir + '/g_weights_after_7k.pth'
# G_image_path = 'data/separator_wo_fibrils.tif'
G_image_path = 'data/nmc_crop.tif'
# D_image_path = 'data/separator_all_grey.tif'
D_image_path = 'data/nmc_crop.tif'

file_name = 'generated_tif.tif'
crop_to_cube = False

# TODO all of these (ngpu, device, to_low_idx, nc_g..) can go into a
#  function in LearnTools that Architecture can also use
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


BM_G = BatchMaker.BatchMaker(path=G_image_path, device=device,
                            rot_and_mir=False)
BM_D = BatchMaker.BatchMaker(path=D_image_path, device=device,
                            rot_and_mir=False)
G_net = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                           scale_factor=scale_f).to(device)
G_net.load_state_dict(torch.load(path_to_g_weights, map_location=torch.device(
    device)))
G_net.eval()


def save_tif_3d(network_g, high_res_im, original_im, grey_idx, device,
                filename,
                mask=False):
    """
        Saves a tif image of the output of G on all of the 3d image high_res_im
    """
    print(high_res_im.size())
    low_res_input = LearnTools.down_sample_for_g_input(high_res_im,
                                                       grey_idx,
                                                       scale_f,
                                                       device, n_dims)
    print(low_res_input.size())

    g_output = network_g(low_res_input).detach().cpu()
    g_output = ImageTools.fractions_to_ohe(g_output)
    g_output_grey = ImageTools.one_hot_decoding(g_output).astype('uint8')
    imsave(progress_main_dir + '/' + filename, g_output_grey)
    low_res_grey = ImageTools.one_hot_decoding(low_res_input.cpu()).astype(
        'uint8')
    imsave(progress_main_dir + '/low_res' + filename,
           low_res_grey)
    high_res_im = ImageTools.one_hot_decoding(original_im.cpu()).astype('uint8')
    imsave(progress_main_dir + '/' + filename + '-original',
           high_res_im)


with torch.no_grad():  # save the images
    im_3d = BM_G.all_image_batch()
    orig_im_3d = BM_D.all_image_batch()
    if crop_to_cube:
        min_d = 128
        im_3d = im_3d[:, :, :min_d, :min_d, :min_d]
        orig_im_3d = orig_im_3d[:, :, :min_d, :min_d, :min_d]
    save_tif_3d(G_net, im_3d, orig_im_3d, to_low_idx, device, file_name,
                mask=False)
    
