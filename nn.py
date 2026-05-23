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
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

    #backward pass
    def backward(self, dvalues):
        #gradient on params
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

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    def forward(self, inputs, training):
        #axis - only along rows (0) or columns (1) -using columns for neuron outputs
        #exponentiate each value in inputs to get unnormalized probs, remove largest to prevent exploding nums
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

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
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

#used for regressors, "squishes" range of outputs betwen 0 and 1
#sigma(i,j) = 1/(1-e^zi,j) z(i,j) = a singular output value of the layer that this activation function takes as input.
#derivative of sigmoid function is just sigma(i,j) * (1 - sigma(i,j)) - check book for proof this is cool
class Activation_Sigmoid:
    #forward pass
    def forward(self, inputs, training):
        #save input, calcute/save outputs of sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
    #backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

#use for regression to predict scalar value rather than class
#y = x , derivative is jsut one
class Activation_Linear:
    #forward
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
    #Backward
    def backward(self, dvalues):
        #1 * dvalues = dvalues
        self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs


class Loss:
    #set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    #regularization loss calc
    #L1 = lambda * sum abs weights
    #l2 = lambda * sum weights **2
    def regularization_loss(self, layer):
        regularization_loss = 0 
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.wxeight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def calculate(self, output, y, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    

class Categorical_Crossentropy(Loss):
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
    def pre_update_params(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update params
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_parames(self, layer):
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
    def pre_update_params(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update params
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_params(self, layer):
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
    def pre_update_params(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update params
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_params(self, layer):
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
    def pre_update_params(self):
        if self.decay:
            #learning rate = learning rate * 1/(1+ decay *iterations) - slowly reduces rate
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    #update params
    #subtract learning rate * parameter gradients - slowly minimizes the loss over time
    def update_params(self, layer):
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
    def forward(self, inputs, training):
        #save inputs 
        self.inputs = inputs
        #generate and save scaled mask - this is equal to a binomial distribution with success rate
        #generates an array with % of zeros = to the rate we want, when we multiply our input array by
        #our binary mask we drop the % of neurons multiplied by those zeros
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
            
    def backward(self, dvalues):
        #gradient on values
        #derivative of the bernoulli distribution is 1/1-q if success, 0 if fail
        self.dinputs = dvalues * self.binary_mask

#generic layer that contains trainging to be used as previous layers for first pass to dense
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

#accuracy class for different accuracy calculations:
class Accuracy:
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        #calculate accuracy
        accuracy = np.mean(comparisons)
        return accuracy
    
class Accuracy_Regression(Accuracy):
    def __init__(self):
        #create precision prop
        self.precision = None
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/250
    #compare predicitions to ground truth
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Classification(Accuracy):
    def __init__(self, *, binary=False):
        #binary or nonbinary classification
        self.binary = binary
    #no initialization needed
    def init(self, y):
        pass
    def compare(self, predictions, y):
        if not(self.binary) and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        #binarymode
        self.binary = binary
    #no initialization needed
    def init(self, y):
        pass
    #compare prediction to ground truth
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    

#make model object for modularity
class Model:

    def __init__(self):
        self.layers = []
        #softmax classifier output object
        self.softmax_classifier_output = None
    
    def add(self, Layer):
        self.layers.append(Layer)
    #set loss and optimizer
    #star means following arguments are keywords, must be passed and no defualts
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    #finalize model
    def finalize(self):
        #create and set input layer
        self.input_layer = Layer_Input()
        #count objest
        layer_count = len(self.layers)
        #create array of trainable layers 
        self.trainable_layers = []
        #iteratie objects
        for i in range(layer_count):
            #if first layer, previous is input, next is just next
            #we're basically making pointers here
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            #else prev is prev, next is next
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            #else last layer next is our loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            #if layer has weights, is trainable
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)
        #if output activation is is Softmax and
        #loss function is categorical cross entropy (check last layer)
        #create an object of combined activation
        #and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Categorical_Crossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    #train the model
    def train(self, X, y, *, epochs=1, print_every = 1, validation_data=None):

        #init accuracy object
        self.accuracy.init(y)
        #main training loop
        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)

            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            self.backward(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')
            if validation_data is not None:
                X_val, y_val = validation_data
                output = self.forward(X_val, training=False)
                loss = self.loss.calculate(output, y_val)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)
                print(f'validation, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}')

    #forward pass/inference
    def forward(self, X, training):
        #call forward on input first to create output that next layers expecting
        self.input_layer.forward(X, training)
        #then call foraward on previous for everything else
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        #last layer
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            #first call backward method on combined activation and loss to set dinputs property
            self.softmax_classifier_output.backward(output, y)
            #set dinputs since not calling backward
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            #call backward method going through all objects but last in reverse order passing dinputs as param
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    


# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
loss=Categorical_Crossentropy(),
optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
accuracy=Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test),
epochs=10000, print_every=100)