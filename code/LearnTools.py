import torch
from torch import autograd
import numpy as np
import ImageTools


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
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    num_images = real_data.size()[0]
    alpha = alpha.expand(batch_size, int(real_data.numel() /
                                         batch_size)).contiguous()
    # print(alpha.shape)
    alpha = alpha.view(num_images, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def down_sample_grey(grey_material):
    """
    :return: a down-sample of the grey material.
    """
    res = torch.nn.AvgPool2d(2, 2)(grey_material)
    res = torch.nn.AvgPool2d(2, 2)(res)
    # threshold at 0.5:
    # return torch.where(res > 0.5, 1., 0.)
    return res


def pixel_wise_distance(low_res_im, generated_high_res, initial_rand):
    """
    calculates and returns the pixel wise distance between the low resolution
    image and the down sampling of the high resolution generated image.
    :return: the normalized distance (divided by the number of pixels of the
    low resolution image
    """
    # since cbd turns into pore in the down-sample, we can just down-sample
    # the grey material
    # ImageTools.show_gray_image(np.array(generated_high_res.detach()[:, 1, :,
    #                                     :])[0, :, :] * 128)
    down_sample = down_sample_grey(generated_high_res[:, 1, :, :])
    low_res_num_pixels = torch.numel(low_res_im[0, 0, :, :])
    # ImageTools.show_gray_image(ImageTools.one_hot_decoding(low_res_im)[0,:,:])
    low_res_grey = low_res_im[:, 1, :, :]
    # ImageTools.show_gray_image(np.array(low_res_grey)[0,:,:]*128)
    # distance is the l2 norm calculating for each image in the batch:
    dist = torch.sum((down_sample-low_res_grey)**2, dim=[1,
                                                         2])/low_res_num_pixels
    # multiplying each image in the batch with the appropriate random number:
    res = torch.mul(dist, initial_rand[:, 0, 0, 0])
    # return the mean:
    return torch.mean(res)

