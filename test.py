import numpy as np
from main import train  # Import the train function from the main module (assuming your main code is in a file named "main.py")

# Set a random seed for reproducibility
np.random.seed(0)

# Define the number of samples
num_samples = 1000

# Define the number of features
num_features = 2

# Generate random input data X
X = np.random.randn(num_features, num_samples)

# Generate random labels Y for binary classification (0 or 1)
Y = np.random.randint(2, size=(1, num_samples))

# Define the architecture of your neural network
layer_dims = [num_features, 4, 1]  # Example architecture with 2 input features, 1 hidden layer with 4 neurons, and 1 output neuron

# Set the number of epochs and learning rate
epochs = 1000  # Number of training iterations
learning_rate = 0.01  # The step size for parameter updates

# Call the train function
trained_params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

# Print the trained parameters (weights and biases)
for key, value in trained_params.items():
    print(f"{key} =\n{value}")

# Print the cost history
print("Cost History:")
for epoch, cost in enumerate(cost_history):
    print(f"Epoch {epoch + 1}: {cost}")
