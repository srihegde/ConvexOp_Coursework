# GradDesc
Simple implementation of training Neural Network to classify MNIST dataset using Gradient Descent.

### Instruction for running the code
Run 'python train_MNIST.py' to check digit classification result.

### About Code Design
We present a flexible code design for generating custom neural network with variable number layers and variable neurons per layer. Also we have presently included implementations of two activation functions - **tanh** and **sigmoid**. 

### Training Procedure
For training our neural network model for classifying digits from MNIST dataset, we consider 3 major phases:
* Forward Pass: Pass each sample through all the layers (from input to output) to produce predicted result.
* Backward Pass: We pass the errors in prediction from output layer to input layer using Backpropagation algorithm.
* Updating Parameters: We update the parameters (both weights and biases) using Gradient Descent updation rule.

This implementation uses **Stochastic Gradient Descent** algorithm - an online algorithm that updates all parameters with passing each data sample through neural network.

### Experiment and Result
In the present implementation we have included a pre-trained neural network model with 1 and 2 hidden layers on MNIST dataset. In both the cases all the hidden layers have 5 neurons. The models are trained with sigmoid activation function on all layers and the loss function we have used is MSE. We have plotted below the results of classifying digit '1' vs all other digits.

![Alt text](./image.png?raw=true "Results based on Accuracy, Precision and Recall for different testing data sizes.")
