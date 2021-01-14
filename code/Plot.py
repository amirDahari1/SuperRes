import os
from matplotlib import pyplot as plt
import matplotlib.image as mpllimg

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/


def plot_img(file_name):
    img_to_plot = mpllimg.imread(file_name)
    plt.imshow(img_to_plot)
    plt.show()


if __name__ == '__main__':
    plot_img('_LossGraph.png')
    plot_img('_GpGraph.png')
    plot_img('fake_slices.png')
