### Time: 5/6/2017
### Authors: Bo Liu, Yunhan Zhao, Cuiqing Li
### Location: department of computer science of the Johns Hopkins University

### Description of the Project:

This is a final project of Data to Model class.  We used Variational Autoencoder to restore the background of photos we collected. For example, in  our model, we have a neural network (feed forward neural nets) which is used to encode our photos into an array, then based on the converted information, we can then create another neural network to restore the photo so that the restored photo is very similair to the original photo (using cross-entropy function and K-L divergence to achieve the target). As for Variational Autoendoer tutorial, you can find an amazing tutorial over [here](http://kvfrans.com/variational-autoencoders-explained/). 

Basically,Variational Autoencoder (VAE) is a type of interesting generative model for unsupervised learning of complicated distributions.Unlike conventional autoencoder, counting on the expressiveness of its neural network encoder and decoder, VAE not only learns a compact hidden representation z for the input X, but also forces z to follow a simple distribution (e.g. a normal distribution) so that X can be easily sampled according to P(X | z).

VAEs are widely used in computer vision. We found the generativity of VAE is particularly interesting and hypothesized that we can take advantage of its generativity to do image completion task. The goal of the Project then is to let VAE learn the generation process of a given type of images, then try to recover the occluded part of images by selecting the most promising reconstructed one. We developed 4 variations of VAEs and test their performance on occluded images from MNIST data.


### Demo of our project poster:
Please click this [link](https://github.com/tiandiao123/Variational-Autoencoder/blob/master/Final_VAE.pdf)!

### Training the Model:
```
python main.py --mode train --loss (BCE, MSE) --model (VAE, VAE_INC, CVAE_LB, CVAE_INC)
```

### Image Autocompletion:
```
python main.py --mode inpainting --loss (BCE, MSE) --model (VAE, VAE_INC, CVAE_LB, CVAE_INC)
```
