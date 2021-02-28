from tifffile import imread, imsave
import numpy as np
import random
import os
import torch
import torch.utils.data
import ImageTools



perms = [[1, 2, 3], [2, 1, 3], [3, 1, 2]]  # permutations for a 4d array.
perms_3d = np.array(perms) + 1
LOW_L_2D = 32  # the low resolution number of pixels LOW_RESxLOW_RES
HIGH_L_2D = 128  # the high resolution number of pixels HIGH_RESxHIGH_RES
N_SAMPLES = 10000
CROP = 4  # crop pixels in each dimension when choosing train slices
LOW_L_3D = 18
HIGH_L_3D = 64

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/
TIF_IMAGE = 'data/NMC.tif'


class BatchMaker:
    """
    Makes and saves training and test batch images.
    """

    def __init__(self, device, path=TIF_IMAGE, dims=3, crop=True):
        """
        :param path: the path of the tif file (TODO make it more general)
        :param dims: number of dimensions for the batches (2 or 3)
        :param device: the device that the image is on.
        :param crop: if to crop the image at the edges
        """
        self.path = path
        self.dims = dims
        self.device = device
        self.im_3d = imread(path)
        self.phases = np.unique(self.im_3d)  # the unique values in image
        self.min_d = min(self.im_3d.shape)  # the minimal dimension
        # crop the image in the edges:
        if crop:
            self.im_3d = self.im_3d[CROP:self.min_d-CROP, CROP:self.min_d-CROP,
                                    CROP:self.min_d-CROP]
            self.min_d = self.min_d - 2*CROP  # update the min dimension
        self.im_ohe = ImageTools.one_hot_encoding(self.im_3d, self.phases)
        if self.dims == 3:
            self.low_l, self.high_l = LOW_L_3D, HIGH_L_3D
        else:  # dims = 2
            self.low_l, self.high_l = LOW_L_2D, HIGH_L_2D
        self.train_scale_factor = self.low_l/self.high_l
        # TODO right now, high_res = 4*low_res -6, make it more general

    def random_batch_for_real(self, batch_size, dim_chosen):
        return self.random_batch2d(batch_size, dim_chosen)

    def random_batch_for_fake(self, batch_size, dim_chosen):
        if self.dims == 3:
            return self.random_batch3d(batch_size, dim_chosen)
        else:  # dims = 2
            return self.random_batch2d(batch_size, dim_chosen)

    def random_batch3d(self, batch_size, dim_chosen):
        """
        :return: A batch of high resolution images,
        along the dimension chosen (0->x,1->y,2->z) in the 3d tif image.
        """
        res = np.zeros((batch_size, len(self.phases),
                        *self.high_l * np.ones(self.dims, dtype=int)))
        for i in range(batch_size):
            res[i, ...] = self.generate_a_random_image3d(dim_chosen)
        # return a torch tensor:
        return torch.FloatTensor(res).to(self.device)

    def generate_a_random_image3d(self, dim_chosen):
        """
        :param dim_chosen: the dimension chosen for the slice
        :return: A random image of size res from the dimension chosen of the
        image. TODO I don't think we can separate between 2d and 3d here
        TODO because of slice
        """
        h_r = self.high_l
        lim_pix = self.min_d - h_r  # the maximum pixel to start with
        # the starting pixels of the other dimensions:
        s_pix = np.random.randint(0, lim_pix, size=self.dims)
        s_x, s_y, s_z = s_pix  # starting voxels, TODO see how to generalise
        # TODO for 2d as well..
        res_image = self.im_ohe[:, s_x:s_x + h_r, s_y:s_y + h_r, s_z:s_z + h_r]
        # for different view, change the cube around..
        return res_image.transpose(0, *perms[dim_chosen])

    def random_batch2d(self, batch_size, dim_chosen):
        """
        :return: A batch of high resolution images, TODO 2d function
        along the dimension chosen (0->x,1->y,2->z) in the 3d tif image.
        """
        res = np.zeros((batch_size, len(self.phases), self.high_l,
                        self.high_l))
        for i in range(batch_size):
            res[i, :, :, :] = self.generate_a_random_image2d(dim_chosen)
        # return a torch tensor:
        return torch.FloatTensor(res).to(self.device)

    def generate_a_random_image2d(self, dim_chosen):
        """
        :param dim_chosen: the dimension chosen for the slice TODO 2d function
        :return: A random image of size res from the dimension chosen of the
        image.
        """
        slice_chosen = random.randint(0, self.min_d - 1)  # the
        # slice chosen
        lim_pix = self.min_d - self.high_l  # the maximum pixel to start with
        # the starting pixels of the other dimensions:
        pix1 = random.randint(0, lim_pix)
        pix2 = random.randint(0, lim_pix)
        if dim_chosen == 0:
            res_image = self.im_ohe[:, slice_chosen, pix1:pix1 + self.high_l,
                                    pix2:pix2 + self.high_l]
        elif dim_chosen == 1:
            res_image = self.im_ohe[:, pix1:pix1 + self.high_l, slice_chosen,
                                    pix2:pix2 + self.high_l]
        else:  # dim_chosen == 2
            res_image = self.im_ohe[:, pix1:pix1 + self.high_l, pix2:pix2 +
                                    self.high_l, slice_chosen]
        return res_image

    def all_image_batch(self, dim, all_image=False):
        """
        :param dim: the dimension to slice the images.
        :param all_image: if True, all image is chosen, if False,
        only middle part of the image at given dimension is chosen with high
        resolution
        :param device: the device that G is on.
        :return: a 3d image with dimension Depthx3xWidthxHeight
        """
        start = 0  # the start pixel
        resolution = self.min_d
        perm = perms[dim]
        if not all_image:
            # s.t. the image will be in the middle
            start = (self.min_d - self.high_l) // 2
            resolution = self.high_l
        if self.dims == 3:
            res = self.im_ohe[:, start:start + resolution, start:start +
                              resolution, start:start + resolution]
            return torch.FloatTensor(res).to(self.device).unsqueeze(0)
        res = self.im_ohe.transpose(perm[0], 0, *perm[1:])
        res = res[:, :, start:start + resolution, start:start + resolution]
        return torch.FloatTensor(res).to(self.device)


def main():
    BM = BatchMaker('cpu')
    cubes = BM.random_batch3d(8, 0)
    print(BM.im_ohe.shape)
    print(cubes.size())


if __name__ == '__main__':
    main()
