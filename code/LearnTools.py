import torch
from torch.nn.functional import interpolate
from torch import autograd

separator = True
k_logistic = 30  # the logistic function coefficient
up_sample_factor = 4
threshold = 0.5
modes = ['bilinear', 'trilinear']


def return_args(parser):
    parser.add_argument('-d', '--directory', type=str, default='default',
                        help='Stores the progress output in the \
                        directory name given')
    parser.add_argument("--squash_phases", default=False, action="store_true",
                        help="All material phases in low res are the same.")
    parser.add_argument('-phases_idx', '--phases_low_res_idx', nargs='+',
                        type=int, default=[1])
    parser.add_argument('-d_dimensions', '--d_dimensions_to_check', nargs='+',
                        type=int, default=[0, 1, 2])
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
    alpha = torch.rand(batch_size, 1, device = device)
    num_images = real_data.size()[0]
    alpha = alpha.expand(batch_size, int(real_data.numel() /
                                         batch_size)).contiguous()
    # print(alpha.shape)
    alpha = alpha.view(num_images, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    # pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def down_sample(high_res_multi_phase, mat_idx, scale_factor, device, n_dims,
                squash=True):
    """
    :param high_res_multi_phase: the high resolution image to downsample.
    :param mat_idx: the indices of phases to downsample.
    :param scale_factor: what is the scale factor of the downsample.
    :param device: cpu or gpu
    :param n_dims: 2d or 3d
    :param squash: if to squash all material phases together for
    downsampling. (when it is hard to distinguish between material phases in
    low resolution e.g. SOFC cathode.)
    :return: a down-sample image of the high resolution image.
    """
    # first choose the material phase in the image:
    material_phases = torch.index_select(high_res_multi_phase, 1, mat_idx)
    if squash:  # all phases of material are same in low-res
        # sum all the material phases:
        material_phases = torch.sum(material_phases, dim=1).unsqueeze(dim=1)
    # down sample:
    material_low_res = interpolate(material_phases, scale_factor=scale_factor,
                                   mode=modes[n_dims - 2])
    # make the pore channel:
    if squash:  # material_low_res already in one channel
        pore_phase = torch.ones(size=material_low_res.size()).to(device) - \
                     material_low_res
    else:  # material_low_res can be in multiple channels
        sum_of_low_res = torch.sum(material_low_res, dim=1).unsqueeze(dim=1)
        pore_phase = torch.ones(size=sum_of_low_res.size()).to(
            device) - sum_of_low_res
    return pore_phase, material_low_res


def down_sample_for_g_input(high_res_multi_phase, mat_idx, scale_factor,
                            device, n_dims, squash=True):
    """
    :return: a down-sample image of the high resolution image for the input
    of G.
    """
    pore_phase, material_low_res = down_sample(high_res_multi_phase, mat_idx,
                                               scale_factor, device, n_dims,
                                               squash)
    # threshold at 0.5:
    material_low_res = torch.where(material_low_res > 0.5, 1., 0.)
    # concat pore and material:
    return torch.cat((pore_phase, material_low_res), dim=1)


def logistic_function(x, k, x0):
    """
    :param x: The input
    :param k: The logistic coefficient
    :param x0: the middle input
    :return: the logistic value of x
    """
    return 1/(1+torch.exp(-k*(x-x0)))


def down_sample_for_similarity_check(generated_im, mat_idx, scale_factor,
                                     device, n_dims, squash=True):
    """
    :return: down sample images of the generated image for the similarity
    check with the low res input, no threshold (logistic function instead)
    for differentiability.
    """
    _, material_low_res = down_sample(generated_im, mat_idx, scale_factor,
                                      device, n_dims, squash)
    return logistic_function(material_low_res, k_logistic, threshold)


def pixel_wise_distance(low_res_im, generated_im, mat_idx,
                        scale_factor, device, n_dims, squash=True):
    """
    calculates and returns the pixel wise distance between the low resolution
    image and the down sampling of the high resolution generated image.
    :return: the normalized distance (divided by the number of pixels of the
    low resolution image
    """
    # all low res phases which are not pore are to be matched:
    low_res_mat = low_res_im[:, 1:]
    down_sample_im = down_sample_for_similarity_check(generated_im, mat_idx,
                                                      scale_factor, device,
                                                      n_dims, squash)
    if separator:  # no punishment for making more material where pore is in
        # low_res
        down_sample_im[low_res_mat == 0] = 0
    return torch.nn.MSELoss()(low_res_mat, down_sample_im)

