import BatchMaker
import LearnTools
import Networks
import ImageTools
import argparse
import torch
import numpy as np
from tifffile import imsave, imread
from torch.nn.functional import interpolate
# import wandb

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash = args.squash_phases
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
size_to_evaluate = args.volume_size_to_evaluate
g_file_name = args.g_image_path
phases_to_low = args.phases_low_res_idx

down_sample = False

progress_main_dir = 'progress/' + progress_dir
# progress_main_dir = 'progress'
path_to_g_weights = progress_main_dir + '/g_weights.pth'
# path_to_g_weights = progress_main_dir + '/g_weights_large_slice.pth'
G_image_path = 'data/' + g_file_name
# G_image_path = 'data/new_vol_down_sample.tif'

file_name = 'generated_tif.tif'
crop_to_cube = False
down_sample_without_memory = args.down_sample
input_with_noise = True
all_pore_input = False

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
    if input_with_noise:
        nc_g = 3
    else:
        nc_g = 2
else:
    if input_with_noise:
        nc_g = 1 + to_low_idx.size()[0] + 1 # channel for pore plus number of
    # material phases to low res.
    else:
        nc_g = 1 + to_low_idx.size()[0]
nc_d = 3  # three phases for the discriminator input TODO somehow make this
# into a flag

G_net = Networks.generator(ngpu, wg, nc_g, nc_d, n_res_blocks, n_dims,
                           scale_factor=scale_f).to(device)
G_net.load_state_dict(torch.load(path_to_g_weights, map_location=torch.device(
    device)))
G_net.eval()


def crop_to_down_sample(high_res):
    """
    If down sample, crops the high resolution image to fit the scale factor.
    """
    dims = np.array(high_res.shape)
    crop_dims = []
    for idx in range(len(dims.shape)):
        dim = dims[idx]
        for subtract in range(dim):
            # doing % twice because the number can be 0 from below (%1.6=1.599)
            if np.round((dim - subtract) % scale_f, 5) % scale_f == 0:
                crop_dims.append(dim - subtract)
                break
    return high_res[:crop_dims[0], :crop_dims[1], :crop_dims[2]]


def down_sample_wo_memory(path):
    high_res_vol = crop_to_down_sample(imread(path))
    ohe_hr_vol = ImageTools.one_hot_encoding(high_res_vol,
                                             np.unique(high_res_vol))
    ohe_hr_vol = np.expand_dims(ohe_hr_vol, axis=0)
    material_phases = torch.index_select(torch.tensor(ohe_hr_vol).to(device), 1,
                                         to_low_idx)
    mat_phase_double = material_phases.double()
    mat_low_res = interpolate(mat_phase_double, scale_factor=1 / scale_f,
                              mode='trilinear')
    mat_low_res += (torch.rand(mat_low_res.size()).to(device) - 0.5) / 100
    mat_low_res = torch.where(mat_low_res > 0.5, 1., 0.)  # TODO to change
    # to ImageTools.fractions_to_ohe also here but more importantly also in
    # the BatchMaker functions of down sampling thresholds.
    sum_of_low_res = torch.sum(mat_low_res, dim=1).unsqueeze(
        dim=1)
    pore_phase = torch.ones(size=sum_of_low_res.size(),
                            device=device) - sum_of_low_res
    return torch.cat((pore_phase, mat_low_res), dim=1)



with torch.no_grad():  # save the images
    # 1. Start a new run
    # wandb.init(project='SuperRes', name='making large volume',
    #            entity='tldr-group')

    step_len = int(np.round(128/scale_f, 5))
    overlap = int(step_len/2)
    high_overlap = int(np.round(overlap / 2 * scale_f, 5))
    step = step_len - overlap

    if down_sample_without_memory:
        im_3d = down_sample_wo_memory(path=G_image_path)
    else:
        BM_G = BatchMaker.BatchMaker(path=G_image_path, device=device,
                                     to_low_idx=to_low_idx, rot_and_mir=False)
        im_3d = BM_G.all_image_batch()

    # orig_im_3d = BM_D.all_image_batch()
    # if crop_to_cube:
    #     min_d = 128
    #     im_3d = im_3d[:, :, :min_d, :min_d, :min_d]
        # orig_im_3d = orig_im_3d[:, :, :min_d, :min_d, :min_d]
    # save_tif_3d(G_net, im_3d, to_low_idx, device, file_name)

    if all_pore_input:
        im_3d[:] = 0
        im_3d[:, 0] = 1

    if input_with_noise:
        input_size = im_3d.size()
        # make noise channel and concatenate it to input:
        noise = torch.randn(input_size[0], 1, *input_size[2:],
                            device=device, dtype=im_3d.dtype)
        im_3d = torch.cat((im_3d, noise), dim=1)

    nz1, nz2, nz3 = size_to_evaluate
    first_img_stack = []
    with torch.no_grad():
        last_ind1 = int(np.ceil((nz1-step_len)/step))
        for i in range(last_ind1 + 1):
            # wandb.log({'large step': i})
            print('i = ' + str(i))
            if i == last_ind1:
                first_lr_vec = im_3d[..., nz1-step_len:nz1, :, :]
            else:
                first_lr_vec = im_3d[..., i*step:i*step+step_len, :, :]
            second_img_stack = []
            last_ind2 = int(np.ceil((nz2-step_len)/step))
            for j in range(last_ind2 + 1):
                print(j)
                if j == last_ind2:
                    second_lr_vec = first_lr_vec[..., :, nz2-step_len:nz2, :]
                else:
                    second_lr_vec = first_lr_vec[..., :, j * step:j * step +
                                                 step_len, :]
                third_img_stack = []
                last_ind3 = int(np.ceil((nz3-step_len)/step))
                for k in range(last_ind3 + 1):
                    # wandb.log({'small step': k})
                    print(k)
                    if k == last_ind3:
                        third_lr_vec = second_lr_vec[..., :, :,
                                       nz3-step_len:nz3]
                    else:
                        third_lr_vec = second_lr_vec[..., :, :, k * step:k *
                                                     step + step_len]
                    g_output = G_net(third_lr_vec).detach().cpu()
                    g_output = ImageTools.fractions_to_ohe(g_output)
                    g_output_grey = ImageTools.one_hot_decoding(
                        g_output).astype('int8').squeeze()
                    if k == 0:  # keep the beginning
                        g_output_grey = g_output_grey[:, :, :-high_overlap]
                    elif k == last_ind3:  # keep the middle+end
                        excess_voxels = int(
                            ((nz3 - step_len) % step) * scale_f)
                        if excess_voxels > 0:
                            g_output_grey = g_output_grey[:, :,
                                                          -(high_overlap +
                                                            excess_voxels):]
                        else:
                            g_output_grey = g_output_grey[:, :, high_overlap:]
                    else:  # keep the middle
                        g_output_grey = g_output_grey[:, :, high_overlap:
                                                      - high_overlap]
                    third_img_stack.append(np.int8(g_output_grey))
                res2 = np.concatenate(third_img_stack, axis=2)
                if j == 0:
                    res2 = res2[:, :-high_overlap, :]
                elif j == last_ind2:
                    excess_voxels = int(((nz2 - step_len) % step) * scale_f)
                    if excess_voxels > 0:
                        res2 = res2[:, -(high_overlap + excess_voxels):, :]
                    else:
                        res2 = res2[:, high_overlap:, :]
                else:
                    res2 = res2[:, high_overlap:-high_overlap, :]
                second_img_stack.append(res2)
            res1 = np.concatenate(second_img_stack, axis=1)
            if i == 0:
                res1 = res1[:-high_overlap, :, :]
            elif i == last_ind1:
                excess_voxels = int(((nz1 - step_len) % step) * scale_f)
                if excess_voxels > 0:
                    res1 = res1[-(high_overlap+excess_voxels):, :, :]
                else:
                    res1 = res1[high_overlap:, :, :]
            else:
                res1 = res1[high_overlap:-high_overlap, :, :]
            first_img_stack.append(res1)
    img = np.concatenate(first_img_stack, axis=0)
    low_res = np.squeeze(ImageTools.one_hot_decoding(im_3d.cpu()))
    if all_pore_input:
        imsave(progress_main_dir + '/' + file_name + '_pore', img)
    else:
        imsave(progress_main_dir + '/' + file_name, img)
    imsave(progress_main_dir + '/' + file_name + 'low_res', low_res)
    # np.save('large_new_vol_nmc', img)