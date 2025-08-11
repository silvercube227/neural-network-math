import numpy as np

#matrix of inputs
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
#matrix of weights
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

#inputs = (3, 4) weights = (4, 3)
#output= input * weight + biases
#transpose weights to turn them into column vectors to dot with the row vectors for matrix product
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outputs)