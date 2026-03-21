#previous files were math notes, building here
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

nnfs.init()

class Layer_Dense:
    #initialize
    #l1 - sum of abs value weights --greater penalties
    #l2 - sum of squared weights --more common
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, 
                 weight_regularizer_l2=0, bias_regularizer_l1=0,
                 bias_regularizer_l2=0):
        #temporarily generate random weights, adjusting the number (0.1) can change the model from learning to not learning - larger number = more variance, more gradient exploding chabce
        #smalkler number = less variance, clustered closer to 0
        self.weights = 0.1  * np.random.randn(n_inputs, n_neurons) #inputs before neurons allows us to avoid transposing this matrix
        #zero bias
        self.biases = np.zeros((1,n_neurons))
        #set regularizer strength 
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    #forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    #backward pass
    def backward(self, dvalues):
        #gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradients on regularization
        #dL1: 1 when >=0, -1 when < 0 - technically 0 undef but that doesnt work
        if self.weight_regularizer_l1 > 0:
            #init ones array
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0 ] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.bias_regularizer_l1 > 0:
            #init ones array
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #dL2 - 2w
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        #axis - only along rows (0) or columns (1) -using columns for neuron outputs
        #exponentiate each value in inputs to get unnormalized probs, remove largest to prevent exploding nums
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        #uninitialized array
        self.dinputs = np.empty_like(dvalues)

        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1, 1)
            #calculate jacobian of output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

#used for regressors, "squishes" range of outputs betwen 0 and 1
#sigma(i,j) = 1/(1-e^zi,j) z(i,j) = a singular output value of the layer that this activation function takes as input.
#derivative of sigmoid function is just sigma(i,j) * (1 - sigma(i,j)) - check book for proof this is cool
class Activation_Sigmoid:
    #forward pass
    def forward(self, inputs):
        #save input, calcute/save outputs of sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    #backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

#use for regression to predict scalar value rather than class
#y = x , derivative is jsut one
class Activation_Linear:
    #forward
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    #Backward
    def backward(self, dvalues):
        #1 * dvalues = dvalues
        self.dinputs = dvalues.copy()


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    #regularization loss calc
    #L1 = lambda * sum abs weights
    #l2 = lambda * sum weights **2
    def regularization_loss(self, layer):
        regularization_loss = 0 
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    

class Categorial_Crossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        #clip both sides to not drag mean toward any val
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        #probabilities for categorical values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        #num samples
        samples = len(dvalues)
        #num labels in sample
        labels = len(dvalues[0])

        #if sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #calculate gradient
        self.dinputs = -y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs/samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Categorial_Crossentropy()
    
    def forward(self, inputs, y_true):
        #output activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #if labels are one-hot (canonical basis vector) encoded, make discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #copy to safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #normalize gradient
        self.dinputs = self.dinputs/samples

#use negative log to compute log likelihood
#sum log likelihoods of correct and incorrect classes for each neuron
#since binary, incorrect = 1- correct
#since model contains multiple binary outputs, each neuron outputs prediction, we need a sample loss 
#sample loss is mean off all losses from single sample - loss from single output is vector of losses for w one val
#for each prediction
class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        #clip data to prevent division by 0 
        #clip both sides to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #calculate sample-wise loss
        sample_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_loss, axis = -1) #-1 tells numpy to calcualte along last dimension
        #return losses
        return sample_losses
    
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs in every sample- use first sample to count
        outputs = len(dvalues[0])
        #clip data to prevent division by 0 
        #clip both sides to not drag mean toward any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        #calculate gradient
        self.dinputs = -(y_true/clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples
    
#for regression, we need to calculate error between predicted scalar and expected scalar
#Use either Mean squared error - average of sum of (y - y^)^2 - pnealize harshly the farther you get
class Loss_MeanSquaredError(Loss): #L2 loss
    #forward pass 
    def forward(self, y_pred, y_true):
        #calculate loss 
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    #backward
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #numberof outputs in every sample
        #use first sample to count
        outputs = len(dvalues[0])
        #gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples

#mean absolute error - average of sum of absolute value of difference between true and predicted
#penalize linearly, sparser results and more robust to outlires
class Loss_MeanAbsoluteError(Loss):
    #forward
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs/samples


#stochastic gradient descent optimizer
class Optimizer_SGD:
    #init optimizer, learning rate of 1
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    #call ohce before any parameter updates
    def pre_update_parameters(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update parameters
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_parameters(self, layer):
        #if using momentum
        if self.momentum:
           #if layre doesn't have momentum arrays, create zero arrays for them
            if not hasattr(layer, 'weight_momentums'):
               layer.weight_momentums = np.zeros_like(layer.weights)
               #also create for biases
               layer.bias_momentums = np.zeros_like(layer.biases)
            #build weight updates with momentum
            #take previous updates multiplied my retain factor (momentum)
            #update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        #normal updates before momentum
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates
    #call once after any param update
    def post_update_params(self):
        self.iterations += 1

#per parameter learning rate
#normalize parameter updates by keeping history of previous updates
#cache stores history of sqrd gradients, parameter updates is fxn of learning rate * gradient
#then divided by sqrt cach + epsilon valie - hyperparam to prevent div by 0
#shrinks gradient descent forever
class Optimizer_AdaGrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    
    #call ohce before any parameter updates
    def pre_update_parameters(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update parameters
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_parameters(self, layer):
        #create cache arrays if we dont have, zero matrix
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        #update cache w squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases**2
        #sgd parameter update +normalize w squared root cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    #call once after any param update
    def post_update_params(self):
        self.iterations += 1

#RMSProp - similar to adagrad, momentum.+ perparameter adaptive leraning rate
#uses moving average of cache, cache contents move with data in real time
#never reach zero - adagrad learning rate goes to zero - stall
#rho - cache memory decay rate - small gradient updates enough to keep going, learning rate is low
class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    #call ohce before any parameter updates
    def pre_update_parameters(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update parameters
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_parameters(self, layer):
        #create cache arrays if we dont have, zero matrix
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        #update cache w squared current gradients - adds the moving average w rho
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases ** 2
        #sgd parameter update +normalize w squared root cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    #call once after any param update
    def post_update_params(self):
        self.iterations += 1

#RMSProp plus the momentum concept from SGD
#apply momentums, add per weight adaptive learning rate
#adds bias correction mechanism, applied to cache and momentumn to compensate for initial zeroed values
#mometum and caches divided by 1 - betastep, as step increase, beta^step approaches 0
#division by fraction causes cache and momemtum to grow faster initially
#replace rho with beta1 and beta 2 for momentum and cache, respectively
#lr 0.001 decay to 0.0001
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    #call ohce before any parameter updates
    def pre_update_parameters(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update parameters
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_parameters(self, layer):
        #create cache arrays and mopmentum arrays if we dont have, zero matrix
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

        #momentum updates - same as sgd except with beta and fraction of beta
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        #correct momentum w division, iterations starts zero, so + 1, 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1)) 
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        #update cache w squared current gradients - adds the moving average w rho
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases ** 2
        #correct our caches
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        #sgd parameter update +normalize w squared root cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    #call once after any param update
    def post_update_params(self):
        self.iterations += 1

class Layer_Dropout:
    #store rate of neurons to dropout = % of neurons we intend to keep
    def __init__(self, rate):
        self.rate = 1 - rate
    #forward pass
    def forward(self, inputs):
        #save inputs 
        self.inputs = inputs
        #generate and save scaled mask - this is equal to a binomial distribution with success rate
        #generates an array with % of zeros = to the rate we want, when we multiply our input array by
        #our binary mask we drop the % of neurons multiplied by those zeros
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        #gradient on values
        #derivative of the bernoulli distribution is 1/1-q if success, 0 if fail
        self.dinputs = dvalues * self.binary_mask

X, y = sine_data()
#dense layer, 2 input features, 64 outputs
#-- for BINARY LOGISTIC REGRESSION --
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

#dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
#dense layer, takes the 64 outputs and converts to 3 outputs
#dense2 = Layer_Dense(512,3)
dense2 = Layer_Dense(64, 64) #blr
#softmax classifier combined loss and activation
#loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#sigmoid activation
activation2 = Activation_ReLU()#blr
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

#Loss
loss_activation = Loss_MeanSquaredError()
#create optimizer
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250

#use loop to perform training, each loop is epoch
for epoch in range(10001):

    dense1.forward(X)
    #take in ouput from previous layer for activation laye
    activation1.forward(dense1.output)
    #forward through dropout 
    #forward pass through second layer, takes output of first layer as input
    dense2.forward(activation1.output)
    #forward pass through sigmoid activation then loss
    activation2.forward(dense2.output) #blr
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_activation.calculate(activation3.output, y)
    #regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    regularization_loss = loss_activation.regularization_loss(dense1) + loss_activation.regularization_loss(dense2) + loss_activation.regularization_loss(dense3) #blr
    loss = data_loss + regularization_loss

    print(f"loss: {loss}")
    #predictions = np.argmax(loss_activation.output, axis=1)
    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = activation3.output #BLS
    #if len(y.shape) == 2:
        #y = np.argmax(y, axis=1)
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')

    #backpropagate 
    loss_activation.backward(activation3.output, y)
    activation3.backward(loss_activation.dinputs) #blr
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #update weights and biases with optimizer
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.update_parameters(dense3)
    optimizer.post_update_params()

#validate model
import matplotlib.pyplot as plt
X_test, y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()

loss = loss_activation.calculate(activation3.output, y_test)
predictions = activation3.output
accuracy = np.mean(np.absolute(predictions - y_test) < accuracy_precision)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')