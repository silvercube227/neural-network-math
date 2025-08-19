#previous files were math notes, building here
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    #initialize
    def __init__(self, n_inputs, n_neurons):
        #temporarily generate random weights
        self.weights = 0.01  * np.random.randn(n_inputs, n_neurons) #inputs before neurons allows us to avoid transposing this matrix
        #zero bias
        self.biases = np.zeros((1,n_neurons))

    #forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        #axis - only along rows (0) or columns (1) -using columns for neuron outputs
        #exponentiate each value in inputs to get unnormalized probs, remove largest to prevent exploding nums
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense1.forward(X)
#take in ouput from previous layer for activation laye
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])