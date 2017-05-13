### Time: 5/6/2017
### Authors: Bo Liu, Yunhan Zhao, Cuiqing Li(Github: tiandiao123)
### Location: department of computer science of the Johns Hopkins University

### Description of the Project:

This is a final project of Data to Model class.  We used Variational Autoencoder to restore the background of photos we collected. For example, in  our model, we have a neural network (feed forward neural nets) which is used to encode our photos into an array, then based on the converted information, we can then create another neural network to restore the photo so that the restored photo is very similair to the original photo (using cross-entropy function and K-L divergence to achieve the target). As for Variational Autoendoer tutorial, you can find an amazing tutorial over [here](http://kvfrans.com/variational-autoencoders-explained/). 

### Training the Model:
```
python main.py --mode train --loss (BCE, MSE) --model (VAE, VAE_INC, CVAE_LB, CVAE_INC)
```

### Image Autocompletion:
```
python main.py --mode inpainting --loss (BCE, MSE) --model (VAE, VAE_INC, CVAE_LB, CVAE_INC)
```
