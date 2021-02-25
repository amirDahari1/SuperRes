import torch
from torch.nn.functional import interpolate
from torch import autograd


k_logistic = 30  # the logistic function coefficient
up_sample_factor = 4
threshold = 0.5


def return_args(parser):
    parser.add_argument('-d', '--directory', type=str, default='default',
                        help='Stores the progress output in the \
                        directory name given')
    parser.add_argument('-wd', '--widthD', type=int, default=8,
                        help='Hyper-parameter for \
                        the width of the Discriminator network')
    parser.add_argument('-wg', '--widthG', type=int, default=8,
                        help='Hyper-parameter for the \
                        width of the Generator network')
    parser.add_argument('-n_res', '--n_res_blocks', type=int, default=2,
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
                        help='The coefficient of the pixel distance loss added '
                             'to the cost of G.')
    args, unknown = parser.parse_known_args()
    return args


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


def down_sample_for_g_input(high_res_3_phase, grey_idx, device):
    """
    :return: a down-sample of the grey material.
    """
    # first choose the grey phase in the image:
    grey_material = torch.index_select(high_res_3_phase, 1, grey_idx)
    # down sample:
    res = torch.nn.AvgPool2d(2, 2)(grey_material)
    res = torch.nn.AvgPool2d(2, 2)(res)
    # threshold at 0.5:
    res = torch.where(res > 0.5, 1., 0.)
    zeros_channel = torch.ones(size=res.size()).to(device) - res
    return torch.cat((zeros_channel, res), dim=1)


def down_sample_for_g_input3d(high_res_3_phase, grey_idx, scale_factor,
                              device):
    """
    :return: a down-sample of the grey material.
    """
    # first choose the grey phase in the image:
    grey_material = torch.index_select(high_res_3_phase, 1, grey_idx)
    # down sample:
    res = interpolate(grey_material, scale_factor=scale_factor,
                      mode='trilinear')  # TODO: maybe different mode?
    # threshold at 0.5:
    res = torch.where(res > 0.5, 1., 0.)
    zeros_channel = torch.ones(size=res.size()).to(device) - res
    return torch.cat((zeros_channel, res), dim=1)


def logistic_function(x, k, x0):
    """
    :param x: The input
    :param k: The logistic coefficient
    :param x0: the middle input
    :return: the logistic value of x
    """
    return 1/(1+torch.exp(-k*(x-x0)))


def down_sample_for_similarity_check(generated_im, grey_idx):
    # first choose the grey phase in the image:
    grey_material = torch.index_select(generated_im, 1, grey_idx)
    # down sample:
    downscale = torch.nn.AvgPool2d(2, 2)
    res = downscale(downscale(grey_material))
    return logistic_function(res, k_logistic, threshold)


def up_sample_for_similarity_check(low_res_im, grey_idx):
    """
    Up-sample the low resolution image G gets as an input for pixel-wise
    similarity with the image generated by G.
    :param low_res_im: G input
    :param grey_idx: index for the grey material
    :return: An up sample (bilinearX4) of the low_res_im
    """
    grey_material = torch.index_select(low_res_im, 1, grey_idx)
    up_sample = torch.nn.Upsample(scale_factor=up_sample_factor,
                                  mode='bilinear')
    return up_sample(grey_material)


def pixel_wise_distance(low_res_im, generated_high_res, grey_idx):
    """
    calculates and returns the pixel wise distance between the low resolution
    image and the down sampling of the high resolution generated image.
    :return: the normalized distance (divided by the number of pixels of the
    low resolution image
    """

    low_res_grey = torch.index_select(low_res_im, 1, grey_idx)
    down_sample = down_sample_for_similarity_check(generated_high_res,
                                                   grey_idx)
    # print(low_res_grey[0, 0, :, :])
    # print(down_sample[0, 0, :, :])
    # print(low_res_grey.size())
    return torch.nn.MSELoss()(low_res_grey, down_sample)

