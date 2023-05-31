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
We have 2 notebooks;  +
The implementation was done with [pytorch lightning](https://www.pytorchlightning.ai/)
* TrainingStyleTransfer.ipynb: This notebook contains some theoretical insights and the code necessary to train 3 different models for Arbitrary Style transfer,
* TestingStyleTransfer.ipynb: This notebook contains some data visualization tools to show the quality of the style transferred images.
####  Check if data is in folder: Step by Step procedure:
    1. Download the data.zip at the url: https://drive.google.com/file/d/1X0QtN8NPjcqJQBFh2Es145z-F75tQvya/view?usp=share_link; 
    2. Insert the uncompressed folder at the same level of this notebook;
    3. Now you can continue to run the cells.   

#### Repository's organization 
* data/ : After the download and the zip extraction, this folder should appear  
  * mscoco/ : MSCoco dataset  
    * train_/ : Training Set (~30k images)  
    * test_/  : Testing Set  (~200 images)  
  * painter_by_numbers/ : Painter by Numbers dataset  
    * train_/ : Training Set (~30k images)  
    * test_/  : Testing Set  (~200 images)  
* models/ : contains 4 models:  
  * vgg_normalised.pth : this is the pre-trained Vgg19 that is common to all the architectures. It was made available at: [https://github.com/naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN.)  
  * vgg.pt : this is the trained Decoder that was obtained in the experiments (vgg19-based).  
  * resnet.pt : this is the trained Decoder that was obtained in the experiments (resnet-based).  
  * resnet_nores.pt : this is the trained Decoder that was obtained in the experiments (resnet-based w/o residuals blocks).   
* paths/ : contains the images used for the experiments, indeed it is possible to augment it to perform novel experiments.  
  * content_i: contains the i-th example of content images;  
  * style_i: contains the i-th example of style images;
* src/ : source code for the experiments  
  * architectures.py : File that contains the implementation of the Encoder, and all the Decoders. The implementation of the Encoder and the Decoder for the Vgg19 was inspired by the implementation at this [repository](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer).    
  * utils.py : file contains some useful function to load, save the models and to visualize the results.






