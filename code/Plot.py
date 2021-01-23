import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpllimg

progress_dir = 'progress/'

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/


def print_eta(filename):
    eta_data = np.load(progress_dir + filename)
    epoch, num_epochs, i, steps, hrs, minutes = eta_data[:]
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, minutes))


def plot_img(file_name):
    img_to_plot = mpllimg.imread(progress_dir + file_name)
    plt.imshow(img_to_plot)
    plt.show()


if __name__ == '__main__':
    print_eta('etaBN.npy')
    plot_img('_LossesGraphBN.png')
    plot_img('_WassGraphBN.png')
    plot_img('_PixelLossBN.png')
    plot_img('_GpGraphBN.png')
    plot_img('fake_slicesBN.png')
