# MNIST-NN

This repository contains a custom implementation of a multilayer perceptron neural network, designed and written from scratch using only numpy. The neural network architecture consists of three layers: an input layer with 784 neurons (ideal for flattened 28x28 pixel images), a hidden layer with 450 neurons, and an output layer with 10 neurons. This makes it particularly well-suited to tackle problems such as digit recognition on the MNIST dataset.

The neural network employs a sigmoid activation function for the first two layers, and a softmax activation function for the output layer. The backpropagation algorithm is utilized for training the network, adjusting both weights and biases via gradient descent.

For training, data is passed in batches, and weights and biases are updated at the end of each batch, which makes the training process more computationally efficient.

The code also includes a prediction function, which makes predictions by feeding the data forward through the network, and a scoring function, which computes the accuracy of the model by comparing the predicted labels to the true labels.

The implementation emphasizes clarity and readability of the code, aiming to provide an educational resource for those who want to delve into the inner workings of neural networks without the abstraction of higher-level libraries. This basic, yet fully functioning, feedforward neural network serves as a solid foundation upon which more complex models and techniques can be built and explored.




