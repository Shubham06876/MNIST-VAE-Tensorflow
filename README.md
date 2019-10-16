# MNIST-VAE-in-Tensorflow
A simple implementation of Variational Autoencoder for MNIST in Tensorflow/Keras.

Original paper: [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) (Diederik P. Kingma, Max Welling).

## Results
### Image reconstruction
A very well-trained VAE model should be able to reconstruct original images.

Here, only the reconstruction of 2D latent dimension is shown. 
 Feel free to try different dimension of latent space.
<tr align='center'>
<td><img src = 'results/reconstruction_images.png' height = '500px'>

### Latent space representation
With latent dimension being 2, the latent space representation can be visualized using 2D scatterplot.
<td><img src = 'results/latent_space_representation.png' height = '500px'>
  
At the same time, the leanred mnist manifold can be visualized as following.
<td><img src = 'results/mnist_manifold.png' height = '500px'>
  
  
## Usage
### Prerequisites
1. Tensorflow
2. Python packages : numpy, scipy, matplotlib, keras.

## References
The implementation is based on the projects: 

[1] https://github.com/wiseodd/generative-models 

[2] https://keras.io/examples/variational_autoencoder/ 

## Acknowledgements
This implementation has been tested with Tensorflow 1.14.0 on macOS Catalina 10.15.
