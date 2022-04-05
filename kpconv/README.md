
![Intro figure](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

This repository contains the modified implementation of **Kernel Point Convolution** (KPConv) in [PyTorch](https://pytorch.org/) and **Unet** in [Pytorch](https://github.com/milesial/Pytorch-UNet)
 
KPConv is a point convolution operator presented in our ICCV2019 paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find our work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

## Installation

### Dependencies
     - numpy
     - scikit-learn
     - PyYAML
     - matplotlib (for visualization)

### Compile c++ extension
Compile the C++ extension modules for python located in `cpp_wrappers`. Open a terminal in this folder, and run:

```bash
          sh compile_wrappers.sh
```

> This implementation has been tested on Ubuntu 18.04 and Windows 10. Details are provided in [INSTALL.md](./INSTALL.md).

