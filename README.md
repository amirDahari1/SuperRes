# SuperRes :telescope:
Welcome to SuperRes! 

To understand the capabilities and theory behind SuperRes, please read our paper:

[Super-resolution of multiphase materials by combining complementary 2D and 3D image data using generative adversarial networks](https://arxiv.org/abs/2110.11281)

## Overview

In very short, SuperRes requires (n >= m): 
- n-phase high-resolution 2D image.
- m-phase low-resolution 3D volume.

To produce an
- n-phase super-resolution 3D volume of the low-resolution 3D volume, with the fine characteristics and added features of the high-resolution 2D image. 

![](paper_figure_for_github.png)

## Usage

### Training
To train the generator, simply run

```
python code/Architecture.py [options] 
```

with the following options:

Option | Type | Default | Description 
--- | --- | --- | ---
-d, --directory | str | 'default' | The name of the directory to save the generator in, under the 'progress' directory.
-sf, --scale_factor | float | 4 | The scale-factor between the high-res slice and low-res volume.
-g_image_path | str | No default | Path to the low-res 3D volume.
-d_image_path | str | No default | Path to the high-res 2D slice. If the material is anisotropic, 3 paths are needed in the correct order.
-phases_idx | int | [1, 2] | The indices of the phases of the low-res input to be compared with the super-res output.
--anisotropic | boolean - stores true | False | Use this option when the material is anisotropic.
--with_rotation | boolean - stores true | False | Use this option for data augmentaion (rotations and mirrors) of the high-res input.

More options are available in ```code/LearnTools.py```

#### Training examples


### Evaluation
To evaluate 




