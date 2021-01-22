from matplotlib import pyplot as plt
import numpy as np
import torch


LOW_RES = 32
HIGH_RES = 128
N_SAMPLES = 10000

progress_dir = 'progress/'


def show_gray_image(image):
    """
    Plots the image in grey scale, assuming the image is 1 channel of 0-255
    """
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def plot_fake_difference(high_res, input_to_g, output_from_g):
    # first move everything to numpy
    rand_sim = np.array(input_to_g[:, 2, :, :])
    images = [high_res, input_to_g[:, :2, :, :], output_from_g]
    images = [np.array(image) for image in images]
    images[2] = fractions_to_ohe(images[2])  # the output from g needs to ohe
    images = [one_hot_decoding(image) for image in images]
    show_three_by_two_gray(images[0], images[1], images[2], rand_sim,
                                      'Very vanilla super-res results')


def show_three_by_two_gray(top_images, middle_images, bottom_images,
                           similarity, title):
    f, axarr = plt.subplots(3, 3)
    axarr[0,0].imshow(top_images[0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[0,1].imshow(top_images[1, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[0,2].imshow(top_images[2, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].imshow(middle_images[0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].imshow(middle_images[1, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 2].imshow(middle_images[2, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].imshow(bottom_images[0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].set_title(str(round(similarity[0, 0, 0].item(), 2)))
    axarr[2, 1].imshow(bottom_images[1, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 1].set_title(str(round(similarity[1, 0, 0].item(), 2)))
    axarr[2, 2].imshow(bottom_images[2, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 2].set_title(str(round(similarity[2, 0, 0].item(), 2)))
    plt.suptitle(title)
    plt.savefig(progress_dir + 'fake_slicesG8D7.png')
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


def down_sample_to_ohe(image):
    """
    :param image: A 3-phase high res image with one-hot-encoding
    :return: a 2-phase low res image with one-hot-encoding with down-sample
    and cbd removal.
    """
    grey_scale_image = one_hot_decoding(image)
    wo_cbd = cbd_to_pore(grey_scale_image)
    down_sample_wo_cbd = down_sample(wo_cbd)
    return torch.FloatTensor(one_hot_encoding(down_sample_wo_cbd))


def one_hot_encoding(image):
    """
    :param image: a [batch_size, height, width] tensor array
    :return: a one-hot encoding of image.
    """
    phases = np.unique(image)  # the unique values in image
    im_shape = image.shape

    res = np.zeros([im_shape[0], len(phases), im_shape[1], im_shape[2]])
    # create one channel per phase for one hot encoding
    for cnt, phs in enumerate(phases):
        image_copy = np.zeros(image.shape)  # just an encoding for one
        # channel
        image_copy[image == phs] = 1
        res[:, cnt, :, :] = image_copy.squeeze()
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
    res = np.zeros([im_shape[0], im_shape[2], im_shape[3]])

    # the assumption is that each pixel has exactly one 1 in its phases
    # and 0 in all other phases:
    for i in range(phases):
        if i == 0:
            continue  # the res is already 0 in all places..
        phase_image = np_image[:, i, :, :]
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


def graph_plot(data, labels, pth, name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(progress_dir + pth + '_' + name)
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