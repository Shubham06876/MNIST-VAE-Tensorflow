# MNIST-VAE-Tensorflow
A simple implementation of Variational Autoencoder for MNIST in Tensorflow/Keras.

Original paper: [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) (Diederik P. Kingma, Max Welling).

## Vanilla VAE
### Image reconstruction
A very well-trained VAE model should be able to reconstruct original images.

Here, only the reconstruction of 2D latent dimension is shown. 
 Feel free to try different dimension of latent space.
<tr align='center'>
<td><img src = 'results/Vanilla VAE/reconstruction_images.png' height = '500px'>

### Latent space representation
With latent dimension being 2, the latent space representation can be visualized using 2D scatterplot.
<td><img src = 'results/Vanilla VAE/latent_space_representation.png' height = '500px'>
  
At the same time, the leanred mnist manifold can be visualized as following.
<td><img src = 'results/Vanilla VAE/mnist_manifold.png' height = '500px'>
 
### FID Score
Here, FID([Fréchet Inception Distance](https://nealjean.com/ml/frechet-inception-distance/)) is used to measured the quality of reconstructed images. 

The lower the FID score, the better the image quality.

After 30 epochs, FID Score = 20.359
 
## Convolutional VAE
Added additional convolutional and drop-out layers.
### Image reconstruction

<tr align='center'>
<td><img src = 'results/Convolutional VAE/reconstruction_images.png' height = '500px'>

### Latent space representation

<td><img src = 'results/Convolutional VAE/latent_space_representation.png' height = '500px'>
  

<td><img src = 'results/Convolutional VAE/mnist_manifold.png' height = '500px'>
 
 ### FID Score

After 100 epochs, FID Score = 15.511
  
  
## Usage
### Prerequisites
1. Tensorflow
2. Python packages : numpy, scipy, matplotlib, keras.

## References
The implementation is based on the projects: 

[1] https://github.com/wiseodd/generative-models 

[2] https://keras.io/examples/variational_autoencoder/ 

[3] https://keras.io/examples/variational_autoencoder_deconv/

## Acknowledgements
This implementation has been tested with Tensorflow 1.14.0 on macOS Catalina 10.15.
