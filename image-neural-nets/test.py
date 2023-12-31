# test.py
from network import Network
from mnist_loader import load_data_wrapper
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the MNIST test data
_, _, test_data = load_data_wrapper()

# Load the pre-trained network parameters (biases and weights)
with open('trained_network.pkl', 'rb') as f:
    network = pickle.load(f)

# Convert the test_data zip object to a list
test_data = list(test_data)

# Visualize sample test
num_samples = 5
fig, axes = plt.subplots(num_samples, 1, figsize=(4, 12))

for i in range(num_samples):
    index = np.random.randint(len(test_data))
    image, label = test_data[index]
    prediction = np.argmax(network.feedforward(image))
    
    # Reshape the image data to its original shape (28x28)
    image = image.reshape((28, 28))
    
    # Plot the image on a subplot
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}, Predicted: {prediction}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Evaluate the network on the entire test data
accuracy = network.evaluate(test_data) / len(test_data) * 100.0 * 10.0
print(f'Test Accuracy: {accuracy:.2f}%')
