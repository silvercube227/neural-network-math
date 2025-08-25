import numpy as np

#matrix of inputs
inputs = [[1.0, 2.0, 3.0, 2.5], #(3,4)
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]
#matrix of weights
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] #(3,4) 
biases = [2.0, 3.0, 0.5] #(3,1)
#inputs = (3, 4) weights = (4, 3)
#output= input * weight + biases
#transpose weights to turn them into column vectors to dot with the row vectors for matrix product
layer_outputs = np.dot(inputs, np.array(weights).T) + biases #(3,3)
#previous layer had 3 neurons, so each weight set must have 3 distinct weights
#3 neurons bc 3 sets of weights and biases
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.74, -0.13]] #(3,3)
biases2 = [[-1, 2, -0.5]] #3,1
#layer outputs become inputs, same transpose for weights and add biases
layer2_outputs = np.dot(layer_outputs, np.array(weights2).T) + biases2 #(3,3)
print(layer_outputs)
print(layer2_outputs)