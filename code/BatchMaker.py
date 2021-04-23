from tifffile import imread, imsave
import numpy as np
import random
import os
import torch
import torch.utils.data
import ImageTools

perms = [[1, 2, 3], [2, 1, 3], [3, 1, 2]]  # permutations for a 4d array.
perms_3d = np.array(perms) + 1 # permutations for a 5d array.
LOW_L_2D = 32  # the low resolution number of pixels LOW_RESxLOW_RES
HIGH_L_2D = 128  # the high resolution number of pixels HIGH_RESxHIGH_RES
CROP = 4  # crop pixels in each dimension when choosing train slices
LOW_L_3D = 45  # length of low resolution 3d
HIGH_L_3D = 64  # length of high resolution 3d

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/
NMC_PATH = 'data/NMC.tif'
SOFC_CATHODE_PATH = 'netl-sofc-cathode-segmented.tif'


class BatchMaker:
    """
    Makes and saves training and test batch images.
    """

    def __init__(self, device, path=NMC_PATH, dims=3, crop=False):
        """
        :param path: the path of the tif file (TODO make it more general)
        :param dims: number of dimensions for the batches (2 or 3)
        :param device: the device that the image is on.
        :param crop: if to crop the image at the edges
        """
        self.path = path
        self.dims = dims  # if G is 3D to 3D or 2D to 2D
        self.device = device
        self.im_3d = imread(path)
        self.dim_im = len(self.im_3d.shape)  # the dimension of the image
        self.phases = np.unique(self.im_3d)  # the unique values in image
        if crop:  # crop the image in the edges:
            self.im_3d = self.im_3d[CROP:-CROP, CROP:-CROP, CROP:-CROP]
        self.im_ohe = ImageTools.one_hot_encoding(self.im_3d, self.phases)
        if self.dims == 3:
            self.low_l, self.high_l = LOW_L_3D, HIGH_L_3D
        else:  # dims = 2
            self.low_l, self.high_l = LOW_L_2D, HIGH_L_2D
        self.scale_factor = self.high_l/self.low_l

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
        # starting voxels
        s_ind = np.random.randint(np.array(self.im_ohe.shape[1:]) - h_r)
        e_ind = s_ind + h_r  # the end indices
        res_image = self.im_ohe[:, s_ind[0]:e_ind[0], s_ind[1]:e_ind[1],
                                s_ind[2]:e_ind[2]]
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
        # the starting pixels of the other dimensions:
        s_ind = np.random.randint(np.array(self.im_ohe.shape[1:]) -
                                  self.high_l)
        e_ind = s_ind + self.high_l
        if self.dim_im == 2:  # the image is just 2D
            return self.im_ohe[:, s_ind[0]:e_ind[0], s_ind[0]:e_ind[0]]
        slice_chosen = np.random.randint(np.array(self.im_ohe.shape[1:]))
        if dim_chosen == 0:
            res_image = self.im_ohe[:, slice_chosen[0], s_ind[1]:e_ind[1],
                                    s_ind[2]:e_ind[2]]
        elif dim_chosen == 1:
            res_image = self.im_ohe[:, s_ind[0]:e_ind[0], slice_chosen[1],
                                    s_ind[2]:e_ind[2]]
        else:  # dim_chosen == 2
            res_image = self.im_ohe[:, s_ind[0]:e_ind[0], s_ind[1]:e_ind[1],
                                    slice_chosen[2]]
        return res_image

    def all_image_batch(self):
        """
        :return: the 3d image ready to be fed into the G with dimensions
        1xCxDxHxW or 1xCxHxW
        """
        return torch.FloatTensor(self.im_ohe).to(self.device).unsqueeze(0)


def main():
    BM = BatchMaker('cpu')
    cubes = BM.random_batch3d(8, 0)
    print(BM.im_ohe.shape)
    print(cubes.size())


if __name__ == '__main__':
    main()
