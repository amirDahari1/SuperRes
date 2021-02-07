from tifffile import imread
import numpy as np
import random
import os
import torch
import torch.utils.data
import ImageTools


LOW_RES = 32
HIGH_RES = 128
N_SAMPLES = 10000

if os.getcwd().endswith('code'):
    os.chdir('..')  # current directory from /SuperRes/code to SuperRes/
TIF_IMAGE = 'data/NMC.tif'


class BatchMaker:
    """
    Makes and saves training and test batch images.
    """

    def __init__(self, path=TIF_IMAGE, n_samples=N_SAMPLES,
                 low_res=LOW_RES, high_res=HIGH_RES):
        """
        :param path: the path of the tif file (TODO make it more general)
        :param n_samples: the number of wanted samples in the batch.
        :param low_res: the low resolution of the 2d image.
        :param high_res: the high resolution of the 2d image.
        """
        self.path = path
        self.im_3d = imread(path)
        self.min_d = min(self.im_3d.shape)  # the minimal dimension of the 3d
        # image
        self.n_samples = n_samples
        self.low_res = low_res
        self.high_res = high_res
        # self.d_train = self.generate_a_random_batch(1)  # test is y slices
        # self.g_train_hr = self.generate_a_random_batch(0)  # train
        # is x slices
        # ImageTools.show_gray_image(self.g_train_hr[0, 0, :, :])
        # self.g_train_no_cbd = ImageTools.cbd_to_grey(self.g_train_hr)
        # self.g_train = ImageTools.down_sample(self.g_train_no_cbd)
        # ImageTools.show_gray_image(self.g_train[0, 0, :, :])
        # change both test and train to one hot encoding:
        # self.ohe_d_train = ImageTools.one_hot_encoding(self.d_train)
        # self.ohe_g_train = ImageTools.one_hot_encoding(self.g_train)
        # self.save_batches()

    def save_batches(self):
        self.ohe_d_train = torch.FloatTensor(self.ohe_d_train)
        dataset = torch.utils.data.TensorDataset(self.ohe_d_train)
        torch.save(dataset, 'data/d_train.pth')
        self.ohe_g_train = torch.FloatTensor(self.ohe_g_train)
        dataset = torch.utils.data.TensorDataset(self.ohe_g_train)
        torch.save(dataset, 'data/g_train.pth')

    def random_batch(self, batch_size, dim_chosen):
        """
        :return: A batch of high resolution images,
        along the dimension chosen (0->x,1->y,2->z) in the 3d tif image.
        """
        res = np.zeros((batch_size, 1, self.high_res, self.high_res))
        for i in range(batch_size):
            res[i, 0, :, :] = self.generate_a_random_image(dim_chosen)
        # one hot encoding:
        res = ImageTools.one_hot_encoding(res)
        # return a torch tensor:
        return torch.from_numpy(res.copy()).type(torch.FloatTensor)

    def generate_a_random_image(self, dim_chosen):
        """
        :param dim_chosen: the dimension chosen for the slice
        :return: A random image of size res from the dimension chosen of the
        image.
        """
        slice_chosen = random.randint(0, self.min_d - 1)  # the slice chosen
        lim_pix = self.min_d - self.high_res  # the maximum pixel to start with
        # the starting pixels of the other dimensions:
        pix1 = random.randint(0, lim_pix)
        pix2 = random.randint(0, lim_pix)
        if dim_chosen == 0:
            res_image = self.im_3d[slice_chosen, pix1:pix1 + self.high_res,
                                   pix2:pix2 + self.high_res]
        elif dim_chosen == 1:
            res_image = self.im_3d[pix1:pix1 + self.high_res, slice_chosen,
                                   pix2:pix2 + self.high_res]
        else:  # dim_chosen == 2
            res_image = self.im_3d[pix1:pix1 + self.high_res, pix2:pix2 +
                                   self.high_res, slice_chosen]
        return res_image


def main():
    BatchMaker()
    # image_3d, min_dim = initialize(tif_images)
    # images = np.zeros((num_samples, 1, IMAGE_2D_HIGH_RES, IMAGE_2D_HIGH_RES),
    #                   dtype=np.uint8)
    # for i in range(num_samples):
    #     images[i, 0, :, :] = generate_a_random_image(image_3d, min_dim, 0)
    #
    # original_tensor = torch.Tensor(images)  # moving the numpy to pytorch
    # # tensors
    # wo_cbd = cbd_to_grey(images)
    # wo_cbd_tensor = torch.Tensor(wo_cbd)
    # small_tensor = down_sample(wo_cbd_tensor)
    # small_numpy = small_tensor.numpy()
    #
    # f, axarr = plt.subplots(3, 3)
    #
    # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    # pic1, pic2, pic3 = 78, 79, 80
    # # just for the picture:
    #
    # axarr[0,0].imshow(images[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    # axarr[1,0].imshow(wo_cbd[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    # axarr[2,0].imshow(small_numpy[pic1,0,:,:], cmap='gray', vmin=0, vmax=255)
    # axarr[0, 1].imshow(images[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    # axarr[1, 1].imshow(wo_cbd[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    # axarr[2, 1].imshow(small_numpy[pic2, 0, :, :], cmap='gray', vmin=0, vmax=255)
    # axarr[0,2].imshow(images[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    # axarr[1,2].imshow(wo_cbd[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    # axarr[2,2].imshow(small_numpy[pic3,0,:,:], cmap='gray', vmin=0, vmax=255)
    # plt.show()
    print('hi')


if __name__ == '__main__':
    main()
