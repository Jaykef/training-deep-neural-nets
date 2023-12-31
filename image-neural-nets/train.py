# train.py

import pickle
from network import Network
from mnist_loader import load_data_wrapper

# Load the MNIST data
training_data, validation_data, test_data = load_data_wrapper()

# Define the network architecture
# For example, a network with 784 input neurons, 30 neurons in the hidden layer, and 10 output neurons
network = Network([784, 30, 10])

# Set hyperparameters
epochs = 30
mini_batch_size = 10
learning_rate = 3.0

# Train the network using stochastic gradient descent
network.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

# Save the trained network parameters (biases and weights) if needed
# For example, using the pickle module: 
with open('trained_network.pkl', 'wb') as f:
    pickle.dump(network, f)
