# Matlab-GAN [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Collection of MATLAB implementations of Generative Adversarial Networks (GANs) suggested in research papers. This repository is greatly inspired by eriklindernoren's repositories [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) and [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN), and contains codes to investigate different architectures of GAN models. 

## Configuration
To run the following codes, users should have the following packages,
- MATLAB 2019b
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional for GPU usage)

## Table of Contents
+ **G**enerative **A**dversarial **N**etwork (GAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/GAN/GAN.m) [[paper]](https://arxiv.org/abs/1406.2661) 
+ **L**east **S**quares **G**enerative **A**dversarial **N**etwork (LSGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/LSGAN/LSGAN.m) [[paper]](https://arxiv.org/abs/1611.04076)
+ **D**eep **C**onvolutional **G**enerative **A**dversarial **N**etwork (DCGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/DCGAN/DCGAN.m) [[paper]](https://arxiv.org/abs/1511.06434)
+ **C**onditional **G**enerative **A**dversarial **N**etwork (CGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/CGAN/CGAN.m) [[paper]](https://arxiv.org/abs/1611.06430)
+ **A**uxiliary **C**lassifier **G**enerative **A**dversarial **N**etwork (ACGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/ACGAN/ACGAN.m) [[paper]](https://arxiv.org/abs/1610.09585)
+ InfoGAN [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/InfoGAN/InfoGAN.m) [[paper]](https://arxiv.org/abs/1606.03657)
+ **A**dversarial **A**uto**E**ncoder (AAE) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/AAE/AAE.m) [[paper]](https://arxiv.org/abs/1511.05644)
+ Pix2Pix [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/Pix2Pix/PIX2PIX.m) [[paper]](https://arxiv.org/abs/1611.07004)
+ **W**asserstein **G**enerative **A**dversarial **N**etwork (WGAN) [[code]](https://github.com/zcemycl/Matlab-GAN/blob/master/WGAN/WGAN.m) [[paper]](https://arxiv.org/abs/1701.07875)
+ **S**emi-Supervised **G**enerative **A**dversarial **N**etwork (SGAN) [[paper]](https://arxiv.org/abs/1606.01583)
+ CycleGAN [[paper]](https://arxiv.org/abs/1703.10593)
+ DiscoGAN [[paper]](https://arxiv.org/abs/1703.05192)

## Outputs
GAN <br>-Generator, Discriminator|  LSGAN <br>-Least Squares Loss | DCGAN <br>-Deep Convolutional Layer | CGAN <br>-Condition Embedding
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/GAN/GANmnist.gif" width="200" > |<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/LSGAN/LSGANresult.jpg" width="200" >|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/DCGAN/DCGANmnist.gif" width="200" >|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/CGAN/CGANmnist.gif" width="200" >
ACGAN <br>-Classification|InfoGAN <br>-Continuous, Discrete Codes|AAE <br>-Encoder, Decoder, Discriminator|Pix2Pix <br>-Pair and Segments checking <br>-Decovolution and Skip Connections
<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/ACGAN/ACGANresult.jpg" width="200"> |<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/InfoGAN/InfoGANmnist.gif" width="200" >|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/AAE/AAEmnist.gif" width="200">|<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/Pix2Pix/p2pfacade.gif" width="200">
WGAN <br>-Energy-based Loss|SGAN|CycleGAN|DiscoGAN
<img src="https://github.com/zcemycl/Matlab-GAN/blob/master/WGAN/resultepoch7.jpg" width="200">|||
