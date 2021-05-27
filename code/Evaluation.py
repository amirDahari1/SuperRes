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
squash = args.squash_phases
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor

phases_to_low = 1
down_sample = False

progress_main_dir = 'progress/' + progress_dir
# progress_main_dir = 'progress'
path_to_g_weights = progress_main_dir + '/g_weights.pth'
# path_to_g_weights = progress_main_dir + '/g_weights_large_slice.pth'
# G_image_path = 'data/separator_wo_fibrils.tif'
G_image_path = 'data/downsample_vol_train_nmc.tif'

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
                            rot_and_mir=False, low_res=not down_sample)
G_net = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                           scale_factor=scale_f).to(device)
G_net.load_state_dict(torch.load(path_to_g_weights, map_location=torch.device(
    device)))
G_net.eval()


def save_tif_3d(network_g, input_to_g, grey_idx, device,
                filename):
    """
        Saves a tif image of the output of G on all of the 3d image high_res_im
    """
    print(input_to_g.size())
    if down_sample:
        input_to_g = LearnTools.down_sample_for_g_input(input_to_g,
                                                       grey_idx,
                                                       scale_f,
                                                       device, n_dims)
    print(input_to_g.size())

    g_output = network_g(input_to_g).detach().cpu()
    g_output = ImageTools.fractions_to_ohe(g_output)
    g_output_grey = ImageTools.one_hot_decoding(g_output).astype('uint8')
    imsave(progress_main_dir + '/' + filename, g_output_grey)
    low_res_grey = ImageTools.one_hot_decoding(input_to_g.cpu()).astype(
        'uint8')
    imsave(progress_main_dir + '/low_res' + filename,
           low_res_grey)
    # high_res_im = ImageTools.one_hot_decoding(original_im.cpu()).astype('uint8')
    # imsave(progress_main_dir + '/' + filename + '-original',
    #        high_res_im)


with torch.no_grad():  # save the images
    im_3d = BM_G.all_image_batch()
    # orig_im_3d = BM_D.all_image_batch()
    # if crop_to_cube:
    #     min_d = 128
    #     im_3d = im_3d[:, :, :min_d, :min_d, :min_d]
        # orig_im_3d = orig_im_3d[:, :, :min_d, :min_d, :min_d]
    # save_tif_3d(G_net, im_3d, to_low_idx, device, file_name)
    
    nz1, nz2, nz3 = 256, 256, 256
    im_3d = im_3d[..., :nz1, :nz2, :nz3].to(device)
    step_len = 32
    overlap = 16
    high_overlap = int(overlap/2 * scale_f)
    step = step_len - overlap
    first_img_stack = []
    with torch.no_grad():
        last_ind1 = int((nz1-step_len)//step)
        for i in range(last_ind1 + 1):
            print('i = ' + str(i))
            first_lr_vec = im_3d[..., i*step:i*step+step_len, :, :]
            second_img_stack = []
            last_ind2 = int((nz2-step_len)//step)
            for j in range(last_ind2 + 1):
                print(j)
                second_lr_vec = first_lr_vec[..., :, j * step:j * step +
                                                             step_len, :]
                third_img_stack = []
                last_ind3 = int((nz3 - step_len) // step)
                for k in range(last_ind3 + 1):
                    print(k)
                    third_lr_vec = second_lr_vec[..., :,
                                                 :, k * step:k * step + step_len]
                    g_output = G_net(third_lr_vec).detach().cpu()
                    g_output = ImageTools.fractions_to_ohe(g_output)
                    g_output_grey = ImageTools.one_hot_decoding(
                        g_output).astype('uint8').squeeze()
                    if k == 0:
                        g_output_grey = g_output_grey[:, :, :-high_overlap]
                    elif k == last_ind3:
                        g_output_grey = g_output_grey[:, :, high_overlap:]
                    else:
                        g_output_grey = g_output_grey[:, :, high_overlap:
                                                      - high_overlap]
                    third_img_stack.append(np.uint8(g_output_grey))
                res2 = np.concatenate(third_img_stack, axis=2)
                if j == 0:
                    res2 = res2[:, :-high_overlap, :]
                elif j == last_ind2:
                    res2 = res2[:, high_overlap:, :]
                else:
                    res2 = res2[:, high_overlap:-high_overlap, :]
                second_img_stack.append(res2)
            res1 = np.concatenate(second_img_stack, axis=1)
            if i == 0:
                res1 = res1[:-high_overlap, :, :]
            elif i == last_ind1:
                res1 = res1[high_overlap:, :, :]
            else:
                res1 = res1[high_overlap:-high_overlap, :, :]
            first_img_stack.append(res1)
    img = np.concatenate(first_img_stack, axis=0)
    imsave(progress_main_dir + '/' + file_name, img)
    # np.save('large_vol_g_2_nmc', img)