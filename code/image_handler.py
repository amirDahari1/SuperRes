from tifffile import imread
from matplotlib import pyplot as plt
import numpy as np
import random
import torch

IMAGE_2D_HIGH_RES = 128
num_samples = 10000

tif_images = '/home/amir/Imperial/images/NMC.tif'

def initialize(path):
    image_3d = imread(tif_images)
    min_d = min(image_3d.shape)  # the minimal dimension of the 3d image
    return image_3d, min_d


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


def generate_a_random_image(image, min_dim, res=IMAGE_2D_HIGH_RES):
    dim_chosen = random.randint(0, 2)  # the dimension to slice
    slice_chosen = random.randint(0, min_dim-1)  # the slice chosen
    lim_pix = min_dim - res  # the maximum pixel to start with
    # the starting pixels of the other dimensions:
    pix1 = random.randint(0, lim_pix)
    pix2 = random.randint(0, lim_pix)
    if dim_chosen == 0:
        res_image = image[slice_chosen, pix1:pix1 + res, pix2:pix2 + res]
    elif dim_chosen == 1:
        res_image = image[pix1:pix1 + res, slice_chosen, pix2:pix2 + res]
    else:  # dim_chosen == 2
        res_image = image[pix1:pix1 + res, pix2:pix2 + res, slice_chosen]
    return res_image


def down_sample(orig_image_tensor):
    """
    Average pool twice, then assigns 255 to values closer to 255 and 0 to
    values closer to 0
    """
    image_tensor = orig_image_tensor.detach().clone()
    image_tensor = torch.nn.AvgPool2d(2, 2)(image_tensor)
    image_tensor = torch.nn.AvgPool2d(2, 2)(image_tensor)
    image_tensor[image_tensor > 64] = 128
    image_tensor[image_tensor <= 64] = 0
    return image_tensor


def cbd_to_grey(orig_image):
    image = np.copy(orig_image)
    image[image == 255] = 0
    return image


def main():
    image_3d, min_dim = initialize(tif_images)
    images = np.zeros((num_samples, 1, IMAGE_2D_HIGH_RES, IMAGE_2D_HIGH_RES),
                      dtype=np.uint8)
    for i in range(num_samples):
        images[i, 0, :, :] = generate_a_random_image(image_3d, min_dim)

    original_tensor = torch.Tensor(images)  # moving the numpy to pytorch
    # tensors
    wo_cbd = cbd_to_grey(images)
    wo_cbd_tensor = torch.Tensor(wo_cbd)
    small_tensor = down_sample(wo_cbd_tensor)
    small_numpy = small_tensor.numpy()

    f, axarr = plt.subplots(3, 3)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    pic1, pic2, pic3 = 78, 79, 80
    # just for the picture:

    axarr[0,0].imshow(images[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    axarr[1,0].imshow(wo_cbd[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    axarr[2,0].imshow(small_numpy[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].imshow(images[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].imshow(wo_cbd[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[2, 1].imshow(small_numpy[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    axarr[0,2].imshow(images[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    axarr[1,2].imshow(wo_cbd[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    axarr[2,2].imshow(small_numpy[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    plt.show()


    print('hi')



if __name__ == '__main__':
    main()
