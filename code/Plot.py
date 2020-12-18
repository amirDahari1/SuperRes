from Architecture import PATH_G, Generator, ngpu, torch, d_dataloader
import ImageTools
import numpy as np

if __name__ == '__main__':
    netG = Generator(ngpu)
    netG.load_state_dict(torch.load(PATH_G))
    high_res = next(iter(d_dataloader))[0]
    print(high_res.shape)
    high_res = ImageTools.one_hot_decoding(high_res)
    print(high_res.shape)
    low_res = ImageTools.cbd_to_grey(high_res)
    low_res = ImageTools.down_sample(low_res)
    low_res = np.expand_dims(low_res, axis=1)
    print(low_res.shape)
    input_to_g = ImageTools.one_hot_encoding(low_res)
    print(low_res.shape)
    fake = netG(torch.FloatTensor(input_to_g)).detach().cpu()
    fake = ImageTools.fractions_to_ohe(fake)
    fake = ImageTools.one_hot_decoding(fake)
    ImageTools.show_three_by_two_gray(high_res, low_res.squeeze(), fake,
                                      'Very vanilla super-res results')