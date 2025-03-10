from matplotlib import pyplot as plt
import numpy as np
import wandb
from taufactor import metrics
from itertools import combinations

LOW_RES = 16
HIGH_RES = 64
N_SAMPLES = 10000

progress_dir = 'progress/'


def show_grey_image(image, title):
    """
    Plots the image in grey scale, assuming the image is 1 channel of 0-255
    """
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    wandb.log({title: [wandb.Image(plt)]})
    # plt.show()


def log_metrics(g_output, hr_metrics):
    """
    Logs the volume fraction and surface area metrics of the
    generated super-res volumes in wandb.
    :param g_output: the current output from g (batch_size*64^3 tensors)
    :param hr_metrics: the same metrics of the high-res 2D slice for
    comparison.
    """
    g_output = one_hot_decoding(fractions_to_ohe(g_output))
    # The super-res volume fraction and surface area values:
    sr_vf, sr_sa = vf_sa_metrics(g_output)
    hr_vf, hr_sa = hr_metrics
    vf_labels, sa_labels = ["VF pore ", "VF AM ", "VF binder "], \
                           ["SA pore/AM ", "SA pore/binder ", "SA AM/binder "]
    [wandb.log({vf_labels[i] + 'SR': sr_vf[i]}) for i in range(len(sr_vf))]
    [wandb.log({vf_labels[i] + 'HR': hr_vf[i]}) for i in range(len(hr_vf))]
    m_loss = [np.abs(1-sr_vf[i]/hr_vf[i]) for i in range(len(hr_vf))]
    [wandb.log({sa_labels[i] + 'SR': sr_sa[i]}) for i in range(len(sr_sa))]
    [wandb.log({sa_labels[i] + 'HR': hr_sa[i]}) for i in range(len(hr_sa))]
    m_loss += [np.abs(1 - sr_sa[i] / hr_sa[i]) for i in range(len(hr_sa))]
    m_loss = np.mean(m_loss)
    wandb.log({'Metrics percentage difference': m_loss})
    # calculate the error in isotropy:
    phases = np.unique(g_output)
    isotropy_errors(g_output, phases[-2], phases[-1])
    return m_loss

def isotropy_errors(batch_images, phase_a, phase_b):
    
    dist = 3
    up_images = np.roll(batch_images, dist, axis=1)
    down_images = np.roll(batch_images, -dist, axis=1)
    right_images = np.roll(batch_images, dist, axis=2)
    left_images = np.roll(batch_images, -dist, axis=2)
    in_images = np.roll(batch_images, dist, axis=3)
    out_images = np.roll(batch_images, -dist, axis=3)
    all_ims = [up_images, down_images, right_images, left_images, in_images, out_images]
    all_ims = [im[:, dist:-dist, dist:-dist, dist:-dist] for im in all_ims]
    str_ims = ['up', 'down', 'right', 'left', 'in', 'out']
    batch_images_cropped = batch_images[:, dist:-dist, dist:-dist, dist:-dist]
    for im, str_im in zip(all_ims, str_ims):
        overlap = ((batch_images_cropped==phase_a) & (im==phase_b)).mean()
        wandb.log({'Isotropy ' + str_im + ' overlap': overlap})

def vf_sa_metrics(batch_images):
    """
    :param batch_images: a 4-dim or 3-dim array of images (batch_size x H x
    W or batch_size x D x H x W)
    :return: a list of the mean volume fractions of the different phases and
    the interfacial surface area between every pair of phases.
    """
    batch_size = batch_images.shape[0]
    phases = np.unique(batch_images)
    vf = np.mean([[(batch_images[j] == p).mean() for p in phases] for j
                  in range(batch_size)], axis=0)
    sa = np.mean([[metrics.surface_area(batch_images[j].astype(np.float32), [ph1, ph2]).item() for
                   ph1, ph2 in combinations(phases, 2)] for j in range(
        batch_size)], axis=0)
    return list(vf), list(sa)


def plot_fake_difference(images, save_dir, filename, with_deg=False):
    # first move everything to numpy
    # rand_sim = np.array(input_to_g[:, 2, :, :])
    images = [np.array(image) for image in images]
    images[1] = fractions_to_ohe(images[1])  # the output from g needs to ohe
    if with_deg:
        images[2] = fractions_to_ohe(images[2])  # also the slices
    images = [one_hot_decoding(image) for image in images]
    save_three_by_two_grey(images, save_dir + ' ' + filename, save_dir,
                           filename, with_deg)


def save_three_by_two_grey(images, title, save_dir, filename, with_deg=False):
    if with_deg:
        f, axarr = plt.subplots(5, 3)
    else:
        f, axarr = plt.subplots(6, 3)
    plane_labels = ['yz', 'xz', 'xy']
    for i in range(3):
        for j in range(3):
            length_im = images[0].shape[1]
            middle_small = int(length_im/2)
            slices = [j]+[slice(None)]*3
            slices[i+1] = middle_small
            axarr[i*2, j].imshow(images[0][tuple(slices)], cmap='gray', vmin=0,
                               vmax=2)
            axarr[i*2, j].set_xticks([0, length_im-1])
            axarr[i*2, j].set_yticks([0, length_im-1])
            # if j == 1:
                # axarr[i*2, j].set_xtitle(f'{plane_labels[i]} plane')
            middle_large = int(images[1].shape[1]/2)
            slices[i+1] = middle_large
            axarr[i*2+1, j].imshow(images[1][tuple(slices)], cmap='gray', vmin=0,
                                 vmax=2)
    if with_deg:
        for j in range(3):  # showing 45 deg slices
            axarr[4, j].imshow(images[3][j, :, :], cmap='gray', vmin=0,
                               vmax=2)
    plt.suptitle(title)
    wandb.log({"running slices": plt})
    plt.savefig(progress_dir + save_dir + '/' + filename + '.png')
    plt.close()


def cbd_to_pore(im_with_cbd):
    """
    :return: the image without cbd. cbd -> pore.
    """
    res = np.copy(im_with_cbd)
    res[res == 255] = 0
    return res


def one_hot_encoding(image, phases):
    """
    :param image: a [depth, height, width] 3d image
    :param phases: the unique phases in the image
    :return: a one-hot encoding of image.
    """
    im_shape = image.shape
    res = np.zeros((len(phases), ) + im_shape, dtype=image.dtype)
    # create one channel per phase for one hot encoding
    for count, phase in enumerate(phases):
        image_copy = np.zeros(im_shape, dtype=image.dtype)  # just an encoding
        # for one channel
        image_copy[image == phase] = 1
        res[count, ...] = image_copy
    return res


def one_hot_decoding(image):
    """
    decodes the image back from one hot encoding to grayscale for
    visualization.
    :param image: a [batch_size, phases, height, width] tensor/numpy array
    :return: a [batch_size, height, width] numpy array
    """
    np_image = np.array(image)
    im_shape = np_image.shape
    phases = im_shape[1]
    res = np.zeros([im_shape[0]] + list(im_shape[2:]))

    # the assumption is that each pixel has exactly one 1 in its phases
    # and 0 in all other phases:
    for i in range(phases):
        if i == 0:
            continue  # the res is already 0 in all places..
        phase_image = np_image[:, i, ...]
        res[phase_image == 1] = i
    return res


def fractions_to_ohe(image):
    """
    :param image: a [n,c,w,h] image (generated) with fractions in the phases.
    :return: a one-hot-encoding of the image with the maximum rule, i.e. the
    phase which has the highest number will be 1 and all else 0.
    """
    np_image = np.array(image)
    res = np.zeros(np_image.shape)
    # Add a little noise for (0.5, 0.5) situations.
    np_image += (np.random.rand(*np_image.shape) - 0.5) / 100
    # finding the indices of the maximum phases:
    arg_phase_max = np.expand_dims(np.argmax(np_image, axis=1), axis=1)
    # make them 1:
    np.put_along_axis(res, arg_phase_max, 1, axis=1)
    return res


def graph_plot(data, labels, pth, filename):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param filename: the directory name to save the plot.
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(progress_dir + pth + '/' + filename)
    plt.close()


def calc_and_save_eta(steps, time, start, i, epoch, num_epochs, filename):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps: number of steps in an epoch
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch: epoch number
    :param num_epochs: total no. of epochs
    :param filename: the filename to save
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    minutes = int((ETA / 3600 % 1) * 60)
    # save_res = np.array([epoch, num_epochs, i, steps, hrs, minutes])
    # np.save(progress_dir + filename, save_res)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, minutes))