# Arbitrary Style Transfer in a data paucity scenario.

The first aim of this project was to reproduce the experiments conducted in the work of [Xun Huang, Serge Belongiein](https://arxiv.org/abs/1703.06868), namely an Encoder-Decoder neural network that is able to perform Style Transfer to a given pair of style and content images.  
The second objective of this project was to test the validity of this Style Transfer approach by using different and more complex nets, w.r.t. the Vgg19, which is that used in the original work.  
The Decoders that were tested are the Resnet34 and its alias network w/o the residual blocks.

## What is style transfer?  
![image](https://drive.google.com/uc?export=view&id=1CyF8m1l-tVChLZmAzqLEKOwp8HbhEMB9)

### This is style transfer.
The heart of their method is in the Adaptive Instance Normalization Layer (AdaIN) that they introduce in the paper.  
All the details can be found and analyzed in this repository, that is organised as follows:  

### How to use this repository:
#### We have 2 notebooks
* TrainingStyleTransfer.ipynb
* TestingStyleTransfer.ipynb  

The former contains the code for the training loop and the explanation of the method adopted. (The implementation was done with [pytorch lightning](https://www.pytorchlightning.ai/))  
The latter contains the code with the experiments done with the different Decoders, and with which is possible to observe the difference performances of them with the aid of some interactive tools.  

* The folder 'paths' contains the images used for the experiments, indeed it is possible to augment it to perform novel experiments.
* The folder 'models' contains 4 models:  
  1. vgg_normalised.pth, this is the pre-trained Vgg19 that is common to all the architectures. It was made available at: [https://github.com/naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN.)  
  2. vgg.pt, this is the trained Decoder that was obtained in the experiments (vgg19-based).  
  3. resnet.pt, this is the trained Decoder that was obtained in the experiments (resnet-based).  
  4. resnet_nores.pt, this is the trained Decoder that was obtained in the experiments (resnet-based w/o residuals blocks).
* The 'architecture.py' file contains the implementation of the Encoder, and all the Decoders. The implementation of the Encoder and the Decoder for the Vgg19 was inspired by the implementation at this [repository](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer).  
* The 'utils.py' file contains some useful function to load, save the models and to visualize the results.
