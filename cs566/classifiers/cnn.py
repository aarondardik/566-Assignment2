from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #Weights will be a 3D tensor. The first index is the
        #number of filter and second indices
        #represent the height and width of each filter
        #this is for the convolutional layer 
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale

        num_channels = input_dim[0]
        #ADDED NUM_CHANNELS BELOW, MAY NEED TO REMOVE THAT AND REDO
        #INDICING OF ANYTHING THAT REFERENCES CONV_SHAPE
        conv_shape = (self.num_filters, num_channels, self.filter_size, self.filter_size)

        #weights = np.zeros((num_filters, filter_size, filter_size))
        self.params['b1'] = np.zeros((num_filters))
        self.params['W1'] = np.random.normal(0.0, self.weight_scale, conv_shape)
        
        input_C = input_dim[0] 
        input_H = input_dim[1] 
        input_W = input_dim[2]

        #Just like in the loss function below, we pick params here
        #such that after convolution, W and H remain the same. 
        #We therefore need W_new = 1 + (W_old - F + 2P)/S = W_old.
        #Thus, if S=1 (there will be multiple possible solutions if
        #we consider the problem purely algebraically, so for ease we fix
        #S=1). Thus, W_old = W_old - F + 2P + 1, i.e. P = (F-1)/2.
        conv_param = {"stride": 1, "pad": (self.filter_size - 1) // 2}

        #after running the conv layer, we have for each input, an output of 
        #shape (num_filters, )


        affine_1_Shape = (self.num_filters, input_H//2, input_W//2)
        #self.params['W2'] = np.random.normal(0.0, self.weight_scale, 
          #affine_1_Shape)

        #WHICH LAYER 2 SHAPE TO USE???? GENUINELY LOOK HERE...
        
        #layer_2_shape = (self.hidden_dim, affine_1_Shape[0]*affine_1_Shape[1]
        #  *affine_1_Shape[2])
        layer_2_shape = (affine_1_Shape[0]*affine_1_Shape[1]*affine_1_Shape[2], 
          self.hidden_dim)


        self.params['W2'] = np.random.normal(0.0, self.weight_scale, layer_2_shape) 
        self.params['b2'] = np.zeros((layer_2_shape[1])) #was previouslly
        #written as np.zeros((self.hidden_dim))...if this doesn't work
        #change b2 back to that


        layer_3_shape = (self.hidden_dim, self.num_classes)
        self.params['W3'] = np.random.normal(0.0, self.weight_scale, layer_3_shape)
        self.params['b3'] = np.zeros((layer_3_shape[1]))

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            #print(type(self.params[k]))
            #print(np.shape(self.params[k]))
            #print(self.params[k][0])
            #print("\n")
            #REMEMBER TO REMOVE PRINT STATEMENTS LATER
        
        

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)


        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size


        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs566/fast_layers.py and  #
        # cs566/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        # Forward pass
        # Convolutional layer + ReLU + Pool
        out, cache_conv_relu_pool = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # Affine layer + ReLU
        #print('Shape of output is: \n')
        #print(np.shape(out))
        #print('passed that')

        out, cache_affine_relu = affine_relu_forward(out, W2, b2)
        #print(np.shape(out))
        #print("passed 2")
        # Final affine layer for class scores
        scores, cache_affine = affine_forward(out, W3, b3)
    

        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute the softmax loss
        loss, dscores = softmax_loss(scores, y)
    
        # Add L2 regularization to the loss
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    
        # Backward pass
        # Backprop into the final affine layer
        dx3, dW3, db3 = affine_backward(dscores, cache_affine)
        grads['W3'] = dW3 + self.reg * W3  # Add regularization gradient
        grads['b3'] = db3
    
        # Backprop into the affine-relu layer
        dx2, dW2, db2 = affine_relu_backward(dx3, cache_affine_relu)
        grads['W2'] = dW2 + self.reg * W2  # Add regularization gradient
        grads['b2'] = db2
    
        # Backprop into the conv-relu-pool layer
        dx1, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_relu_pool)
        grads['W1'] = dW1 + self.reg * W1  # Add regularization gradient
        grads['b1'] = db1



        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
