# Matlab-GAN [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Collection of MATLAB implementations of Generative Adversarial Networks (GANs) suggested in research papers. This repository is greatly inspired by eriklindernoren's repositories [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) and [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN), and contains codes to investigate different architectures of GAN models. 

## Configuration
To run the following codes, users should have the following packages,
- MATLAB 2019b
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional for GPU usage)

## Table of Contents
+ Generative Adversarial Network (GAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/GAN/GAN.m) [[paper]](https://arxiv.org/abs/1406.2661) 
+ Least Squares Generative Adversarial Network (LSGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/LSGAN/LSGAN.m) [[paper]](https://arxiv.org/abs/1611.04076)
+ Deep Convolutional Generative Adversarial Network (DCGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/DCGAN/DCGAN.m) [[paper]](https://arxiv.org/abs/1511.06434)
+ Conditional Generative Adversarial Network (CGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/CGAN/CGAN.m) [[paper]](https://arxiv.org/abs/1611.06430)
+ Auxiliary Classifier Generative Adversarial Network (ACGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/ACGAN/ACGAN.m) [[paper]](https://arxiv.org/abs/1610.09585)
+ InfoGAN [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/InfoGAN/InfoGAN.m) [[paper]](https://arxiv.org/abs/1606.03657)
+ Adversarial Autoencoder 
+ CycleGAN
+ DiscoGAN
+ Pix2Pix

## Outputs
GAN |  LSGAN  | DCGAN | CGAN
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/GAN/GANmnist.gif" width="200" > |<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/LSGAN/LSGANresult.jpg" width="200" >|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/DCGAN/DCGANmnist.gif" width="200" >|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/CGAN/CGANmnist.gif" width="200" >
ACGAN|InfoGAN|Adversarial Autoencoder|CycleGAN
<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/ACGAN/ACGANresult.jpg" width="200"> |<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/InfoGAN/InfoGANmnist.gif" width="200" >||
DiscoGAN|Pix2Pix
