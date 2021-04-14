from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb

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


def plot_fake_difference(high_res, input_to_g, output_from_g,
                         slices_45, save_dir, filename):
    # first move everything to numpy
    # rand_sim = np.array(input_to_g[:, 2, :, :])
    images = [high_res, input_to_g, output_from_g, slices_45]
    images = [np.array(image) for image in images]
    images[2] = fractions_to_ohe(images[2])  # the output from g needs to ohe
    images[3] = fractions_to_ohe(images[3])  # also the slices
    images = [one_hot_decoding(image) for image in images]
    save_three_by_two_grey(images[0], images[1], images[2], images[3],
                           save_dir + ' ' + filename, save_dir, filename)


def save_three_by_two_grey(top_images, middle_images, bottom_images,
                           slices_45, title, save_dir, filename):
    f, axarr = plt.subplots(5, 3)
    images = [top_images, middle_images, bottom_images, slices_45]
    axarr[0,0].imshow(images[0][0, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[0,1].imshow(images[0][1, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[0,2].imshow(images[0][2, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].imshow(images[1][0, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].imshow(images[1][1, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 2].imshow(images[1][2, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].imshow(images[2][0, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 1].imshow(images[2][1, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 2].imshow(images[2][2, 4, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[3, 0].imshow(images[3][0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[3, 1].imshow(images[3][1, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[3, 2].imshow(images[3][5, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[4, 0].imshow(images[2][0, :, :, 4], cmap='gray', vmin=0, vmax=255)
    axarr[4, 1].imshow(images[2][1, :, :, 4], cmap='gray', vmin=0, vmax=255)
    axarr[4, 2].imshow(images[2][2, :, :, 4], cmap='gray', vmin=0, vmax=255)
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


def down_sample(orig_image_tensor):
    """
    Average pool twice, then assigns 128 to values closer to 128 and 0 to
    values closer to 0 (Assumes 2 phases!)
    """
    max_im = np.max(np.array(orig_image_tensor))
    image_tensor = torch.FloatTensor(np.copy(orig_image_tensor))
    image_tensor = torch.nn.AvgPool2d(2, 2)(image_tensor)
    image_tensor = torch.nn.AvgPool2d(2, 2)(image_tensor)
    # threshold in the middle - arbitrary choice
    image_array = np.array(image_tensor)
    image_array[image_array > max_im/2] = max_im
    image_array[image_array <= max_im/2] = 0
    return torch.FloatTensor(image_array)


def one_hot_encoding(image, phases):
    """
    :param image: a [depth, height, width] 3d image
    :param phases: the unique phases in the image
    :return: a one-hot encoding of image.
    """
    im_shape = image.shape
    res = np.zeros((len(phases), ) + im_shape)
    # create one channel per phase for one hot encoding
    for count, phase in enumerate(phases):
        image_copy = np.zeros(im_shape)  # just an encoding for one
        # channel
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
    decodes = [0, 128, 255]
    res = np.zeros([im_shape[0]] + list(im_shape[2:]))

    # the assumption is that each pixel has exactly one 1 in its phases
    # and 0 in all other phases:
    for i in range(phases):
        if i == 0:
            continue  # the res is already 0 in all places..
        phase_image = np_image[:, i, ...]
        res[phase_image == 1] = decodes[i]
    return res


def fractions_to_ohe(image):
    """
    :param image: a [n,c,w,h] image (generated) with fractions in the phases.
    :return: a one-hot-encoding of the image with the maximum rule, i.e. the
    phase which has the highest number will be 1 and all else 0.
    """
    np_image = np.array(image)
    res = np.zeros(np_image.shape)
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
    save_res = np.array([epoch, num_epochs, i, steps, hrs, minutes])
    np.save(progress_dir + filename, save_res)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, minutes))