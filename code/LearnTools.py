import matplotlib.pyplot as plt
import torch
import ImageTools
from torch import nn
from torch.nn import functional
from torch.nn.functional import interpolate
from torch.nn.functional import one_hot
from torch import autograd
import numpy as np
import math  # just so I don't use numpy by accident

k_logistic = 30  # the logistic function coefficient
threshold = 0.5
modes = ['bilinear', 'trilinear']


def return_args(parser):
    parser.add_argument('-d', '--directory', type=str, default='default',
                        help='Stores the progress output in the \
                        directory name given')
    parser.add_argument('-sf', '--scale_factor', type=float, default=4,
                        help='scale factor between high res and low res.')
    parser.add_argument("--down_sample", default=False, action="store_true",
                        help="Down samples the input for G for testing "
                             "purposes.")
    parser.add_argument("--super_sampling", default=False,
                        action="store_true", help="When comparing super-res "
                        "and low-res, instead of blurring, it picks one voxel "
                        "with nearest-neighbour interpolation.")
    parser.add_argument("--squash_phases", default=False, action="store_true",
                        help="All material phases in low res are the same.")
    parser.add_argument("--anisotropic", default=False, action="store_true",
                        help="The material is anisotropic (requires dif Ds).")
    parser.add_argument("--with_rotation", default=False, action="store_true",
                        help="create rotations and mirrors for the BM.")
    parser.add_argument("--separator", default=False, action="store_true",
                        help="Different voxel-wise loss for separator "
                             "material.")
    parser.add_argument('-rotations_bool', nargs='+', type=int,
                        default=[0, 0, 1], help="If the material is "
                        "anisotropic, specify which images can be augmented "
                                                "(rotations and mirrors)")
    parser.add_argument('-g_image_path', type=str, help="Path to the LR "
                        "3D volume")
    parser.add_argument('-d_image_path', nargs='+', type=str, help="Path to "
                        "the HR 2D slice, if Isotropic, 3 paths are needed, "
                        "in correct order")
    parser.add_argument('-phases_idx', '--phases_low_res_idx', nargs='+',
                        type=int, default=[1, 2])
    parser.add_argument('-d_dimensions', '--d_dimensions_to_check', nargs='+',
                        type=int, default=[0, 1, 2])
    parser.add_argument('-volume_size_to_evaluate', nargs='+', type=int,
                        default=[128, 128, 128])
    parser.add_argument('-wd', '--widthD', type=int, default=9,
                        help='Hyper-parameter for \
                        the width of the Discriminator network')
    parser.add_argument('-wg', '--widthG', type=int, default=8,
                        help='Hyper-parameter for the \
                        width of the Generator network')
    parser.add_argument('-n_res', '--n_res_blocks', type=int, default=1,
                        help='Number of residual blocks in the network.')
    parser.add_argument('-n_dims', '--n_dims', type=int, default=3,
                        help='The generated image dimension (and input '
                             'dimension), can be either 2 or 3.')
    parser.add_argument('-gu', '--g_update', type=int, default=5,
                        help='Number of iterations the generator waits before '
                             'being updated')
    parser.add_argument('-e', '--num_epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('-pix_d', '--pixel_coefficient_distance', type=int,
                        default=10,
                        help='The coefficient of the pixel distance loss '
                             'added to the cost of G.')
    args, unknown = parser.parse_known_args()
    return args


def forty_five_deg_masks(batch_size, phases, high_l):
    """
    :param batch_size: batch size for the images for the making of the mask.
    :param phases: number of phases.
    :param high_l: the length of the high resolution
    :return: list of two masks of the 45 degree angle slices along the
    z-axis of the 3d (returns masks for both slices of 45 degrees).
    """
    over_sqrt_2 = int(high_l/math.sqrt(2))  # high_l in the diagonal

    # create the masks:
    masks = []
    for m in range(high_l - over_sqrt_2):
        mask1 = torch.zeros((batch_size, phases, *[high_l] * 3),
                            dtype=torch.bool)
        mask2 = torch.zeros(mask1.size(), dtype=torch.bool)
        mask3 = torch.zeros(mask1.size(), dtype=torch.bool)
        mask4 = torch.zeros(mask1.size(), dtype=torch.bool)
        if m == 0:
            for i in range(over_sqrt_2):
                mask1[..., m + i, i, :] = True
                mask3[..., i + m, -1 - i, :] = True
            masks.extend([mask1, mask3])
        else:
            for i in range(over_sqrt_2):
                mask1[..., m + i, i, :] = True
                mask2[..., i, m + i, :] = True
                mask3[..., i + m, -1 - i, :] = True
                mask4[..., i, -1 - (i + m), :] = True
            masks.extend([mask1, mask2, mask3, mask4])
    return masks


def to_slice(k, forty_five_deg, D_dimensions_to_check):
    """
    :param k: axis idx.
    :param forty_five_deg: bool determining if to slice in 45 deg.
    :param D_dimensions_to_check: The dimensions to check by the user.
    :return: When to slice the volume (in which axis/45 deg angles).
    """
    if k not in D_dimensions_to_check:
        if k != 2:
            return False
        if not forty_five_deg:
            return False
    return True


def forty_five_deg_slices(masks, volume_input):
    """
    :param masks: the masks of the 45 degree angle slices
    :param volume_input: the volume to slice
    :return: the two slices (as a tensor of batch size x 2)
    """
    tensors = []
    batch_size, phases, high_l = volume_input.size()[:3]
    for mask in masks:
        # the result of the mask on the input:
        slice_mask = volume_input[mask].view(batch_size, phases, -1, high_l)
        # add the slice after up_sample to wanted size:
        tensors.append(interpolate(slice_mask, size=(high_l, high_l), mode=modes[0]))
    return torch.cat(tensors, dim=0)  # concat tensors along batch_size


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device,
                          gp_lambda, nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    # sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device=device)
    num_images = real_data.size()[0]
    alpha = alpha.expand(batch_size, int(real_data.numel() /
                                         batch_size)).contiguous()
    alpha = alpha.view(num_images, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    # pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(),
                                                      device=device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


class DownSample(nn.Module):
    """
    Calculates the down-sampled version of the generated volume. Can also be
    used to generate a low-res volume from a high-res volume for evaluation
    reasons.
    """
    def __init__(self, squash, n_dims, low_res_idx, scale_factor,
                 device, super_sampling=False, separator=False):
        """
        :param n_dims: 2d to 2d or 3d to 3d.
        :param low_res_idx: the indices of phases to down-sample.
        :param scale_factor: scale factor between high-res and low-res.
        :param squash: if to squash all material phases together for
        :param separator: different voxel-wise loss for the separator material.
        :param device: The device the object is on.
        down-sampling. (when it is hard to distinguish between material phases
        in low resolution e.g. SOFC cathode.)
        """
        super(DownSample, self).__init__()
        self.squash = squash
        self.n_dims = n_dims
        # Here we want to compare the pore as well:
        self.low_res_idx = torch.cat((torch.zeros(1).to(low_res_idx),
                                      low_res_idx))

        self.low_res_len = self.low_res_idx.numel()  # how many phases
        self.scale_factor = scale_factor
        self.device = device
        self.separator = separator
        self.voxel_wise_loss = nn.MSELoss()  # the voxel-wise loss
        # Calculate the gaussian kernel and make the 3d convolution:
        self.gaussian_k = self.calc_gaussian_kernel_3d(self.scale_factor)
        # Reshape to convolutional weight
        self.gaussian_k = self.gaussian_k.view(1, 1, *self.gaussian_k.size())
        self.gaussian_k = self.gaussian_k.repeat(self.low_res_len, *[1] * (
                self.gaussian_k.dim() - 1)).to(self.device)
        self.groups = self.low_res_len  # ensures that each phase will be
        # blurred independently.
        self.gaussian_conv = functional.conv3d
        self.softmax = functional.softmax
        self.super_sampling = super_sampling

    def voxel_wise_distance(self, generated_im, low_res):
        """
        calculates and returns the pixel wise distance between the low-res
        image and the down sampling of the high-res generated image.
        :return: the normalized distance (divided by the number of pixels of
        the low resolution image.)
        """
        down_sampled_im = self(generated_im)
        if self.separator:  # no punishment for making more material where pore
            # is in low_res. All low res phases which are not pore are to be
            # matched:
            low_res = low_res[:, 1:]
            down_sampled_im = down_sampled_im[:, 1:]
            down_sampled_im = down_sampled_im * low_res
            return torch.nn.MSELoss()(low_res, down_sampled_im)
        # There is a double error for a mismatch:
        mse_loss = torch.nn.MSELoss()(low_res, down_sampled_im)
        return mse_loss * self.low_res_len / 2  # to standardize the loss.

    def forward(self, generated_im, low_res_input=False):
        """
        Apply gaussian filter to the generated image.
        """
        # First choose the material phase in the image:
        low_res_phases = torch.index_select(generated_im, 1, self.low_res_idx)
        if self.squash:  # all phases of material are same in low-res
            # sum all the material phases:
            low_res_phases = torch.sum(low_res_phases, dim=1).unsqueeze(
                dim=1)
        # if it is super-sampling, return nearest-neighbour interpolation:
        if self.super_sampling:
            return interpolate(low_res_phases, scale_factor=1 /
                               self.scale_factor, mode='nearest')
        # Then gaussian blur the low res phases generated image:
        blurred_im = self.gaussian_conv(input=low_res_phases,
                                        weight=self.gaussian_k,
                                        padding='same', groups=self.groups)
        # Then downsample using trilinear interpolation:
        blurred_low_res = interpolate(blurred_im,
                                      scale_factor=1 / self.scale_factor,
                                      mode=modes[self.n_dims - 2])
        if low_res_input:  # calculate a low-res input.
            return self.get_low_res_input(blurred_low_res)
        # Multiplying the softmax probabilities by a large number to get a
        # differentiable argmax function to avoid blocky super-res volumes:
        return self.softmax(blurred_low_res*100, dim=1)

    def get_low_res_input(self, blurred_image):
        """
        If only the low-res input is to be calculated for evaluation study.
        :param blurred_image: after the image has been blurred and
        down-sampled.
        :return: a batch_size X low_res_phases X *low_res_vol_dimensions of a
        for a one-hot volume.
        """
        # Adding little noise for the (0.5, 0.5) scenarios.
        blurred_image += (torch.rand(blurred_image.size(),
                                     device=blurred_image.device) - 0.5) / 1000
        num_phases = blurred_image.size()[1]
        blurred_image = torch.argmax(blurred_image, dim=1)  # find max phase
        one_hot_vol = one_hot(blurred_image, num_classes=num_phases)
        return one_hot_vol.permute(0, -1, *torch.arange(1, self.n_dims + 1))

    @staticmethod
    def calc_gaussian_kernel_3d(scale_factor):
        """
        :param scale_factor: The scale factor used between the low- and
        high-res volumes.
        :return: A gaussian blur 3d kernel for blurring before interpolating
        """
        ks = math.ceil(scale_factor)  # the kernel size
        if ks % 2 == 0:
            ks -= 1  # if even, the closest odd number from below.
        # The same default sigma as in transforms.functional.gaussian_blur:
        sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
        ts = torch.linspace(-(ks // 2), ks // 2, ks)
        gauss = torch.exp((-(ts / sigma) ** 2 / 2))
        kernel_1d = gauss / gauss.sum()  # Normalization
        # 3d gaussian kernel can be computed in the following way:
        kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)
        return kernel_3d


# if __name__ == '__main__':
#     downsample_test = DownSample(squash=False, n_dims=3,
#                                  low_res_idx=torch.LongTensor([1, 2, 3]),
#                                  scale_factor=4)
#     gen_im = torch.zeros(1, 5, 4, 4, 4)
#     gen_im[0, 1, 2:,2:,2:] = 1
#     gen_im[0, 2, :2,:2,:2] = 1
#     low_res = torch.zeros(1, 4, 1, 1, 1)
#     low_res[0, 2] = 1
#     res1 = downsample_test(gen_im)
#     res2 = downsample_test(gen_im, low_res_input=True)
#     loss = downsample_test.voxel_wise_distance(gen_im, low_res)

