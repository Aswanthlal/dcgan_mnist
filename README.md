# DCGAN on MNIST

### This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) in a single Python script using TensorFlow 2. The model learns to generate handwritten digit images similar to the MNIST dataset.

* Complete GAN training loop in one Python file

* CNN-based Generator and Discriminator

* Uses tf.data.Dataset pipeline for batching MNIST data

* Saves image snapshots and a .gif of generated samples over epochs

* Checkpointing for saving and resuming training


## Model Details
### Generator:
* Input: 100-dim noise vector

* Architecture: Dense → Reshape → Conv2DTranspose ×3

* Output: 28×28×1 image (tanh activation)

### Discriminator:
* Input: 28×28×1 image

* Architecture: Conv2D ×2 → Flatten → Dense

* Output: Real/Fake probability (from_logits=True)


### Training Setup
* Optimizer: Adam (1e-4)

* Batch Size: 256

* Loss: Binary Cross-Entropy

* Epochs: 100


### Notes
* Output is normalized to [-1, 1] for compatibility with tanh activation

* Fixed seed used for consistent image generation over time

* Checkpoints saved every 15 epochs
