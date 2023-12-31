from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        pass
    
        mu = 0
        
        self.params["W1"] = np.random.normal(mu, weight_scale, (input_dim, hidden_dim)) # create weights for the first layer
        self.params["b1"] = np.zeros(hidden_dim)
        
        self.params["W2"] = np.random.normal(mu, weight_scale, (hidden_dim, num_classes)) # Create weights for the second layer
        self.params["b2"] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        
        L1_out, L1_cache = affine_forward(X, self.params["W1"], self.params["b1"])
        
        A1, A1_cache = relu_forward(L1_out)
        
        L2_out, L2_cache = affine_forward(A1, self.params["W2"], self.params["b2"])
        
        scores = L2_out
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass 
        
        loss, dx = softmax_loss(scores, y)
        
        loss += 0.5*self.reg*(np.sum(np.square(self.params["W1"])) + np.sum(np.square(self.params["W2"]))) # Loss L2 regularization
        
        dx2, grads["W2"], grads["b2"] = affine_backward(dx, L2_cache)
        
        dx3 = relu_backward(dx2, A1_cache)
        
        dx4, grads["W1"], grads["b1"] = affine_backward(dx3, L1_cache)
        
        grads["W2"] += self.reg*self.params["W2"] # Gradient regularization (simply lambda*W)
        grads["W1"] += self.reg*self.params["W1"]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims) # This is L
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        pass
        
        mu = 0 # mean = 0
        
        for i in range(self.num_layers - 1): # Loop over the layers (-1 is to account for the last layer). Note i goes from 0 -> self.num_layers - 2 I.E The last number is never hit
            if i == 0: # note that the loop index starts at 0, this if statement is to catch the first case which has different input_dim
                self.params['W1'] = np.random.normal(mu, weight_scale, (input_dim, hidden_dims[i])) # weights are taken from normal dist
                self.params['b1'] = np.zeros((hidden_dims[i])) # b is initialized to zeros
                if normalization == "batchnorm":
                    self.params['gamma1'] = np.ones(hidden_dims[i]) # Note that batchnorm will take the dimensions of the next layer
                    self.params['beta1'] = np.zeros(hidden_dims[i]) # Since it works on the next layers neurons
                    
            else: # The rest of the things
                current_weight = 'W' + str(i+1)
                self.params[current_weight] = np.random.normal(mu, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
                current_bias = 'b' + str(i+1)
                self.params[current_bias] = np.zeros(hidden_dims[i])
                
                if normalization == "batchnorm":
                    self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])
        
        # This is for the final case
        current_weight = 'W' + str(self.num_layers)
        self.params[current_weight] = np.random.normal(mu, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        current_bias = 'b' + str(self.num_layers)
        self.params[current_bias] = np.zeros(num_classes)
        
        """
        if normalization == "batchnorm":
            self.params['gamma' + str(self.num_layers)] = np.ones(hidden_dims[i])
            self.params['beta' + str(self.num_layers)] = np.zeros(hidden_dims[i])
        """
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass
        
        self.outputs = {} # To store the intermediate outputs
        # The number of hidden layers is num_layers - 1 to account for the final affine layer
        
        # Loop over all hidden layers performing the forward pass
        for i in range(self.num_layers - 1):
            if i == 0: # For the first layer where we cannot properly index X within the for loop
                self.outputs['L1_out'], self.outputs['L1_cache'] = affine_forward(X, self.params['W1'], self.params['b1'])
                
                # If there is batch normalization then we inject it between the layers. Otherwise we just do relu
                if self.normalization == "batchnorm":
                    self.outputs['B1_out'], self.outputs['B1_cache'] = batchnorm_forward(self.outputs['L1_out'], self.params['gamma1'], self.params['beta1'], self.bn_params[i])
                    self.outputs['A1_out'], self.outputs['A1_cache'] = relu_forward(self.outputs['B1_out'])
                else:
                    self.outputs['A1_out'], self.outputs['A1_cache'] = relu_forward(self.outputs['L1_out'])
                    
                if self.use_dropout:
                    self.outputs['A1_out'], self.outputs['D1_cache'] = dropout_forward(self.outputs['A1_out'], self.dropout_param)
                    
            else: # For the rest of the hidden layers
                
                current_weight = 'W' + str(i+1)
                current_bias = 'b' + str(i+1)
                current_affine_output = 'L' + str(i+1) + '_out'
                current_affine_cache = 'L' + str(i+1) + '_cache'
                current_ReLU_output = 'A' + str(i+1) + '_out'
                current_ReLU_cache = 'A' + str(i+1) + '_cache'
                previous_ReLU_output = 'A' + str(i) + '_out'
                
                self.outputs[current_affine_output], self.outputs[current_affine_cache] = affine_forward(self.outputs[previous_ReLU_output], self.params[current_weight], self.params[current_bias])
                
                if self.normalization == "batchnorm":
                    self.outputs['B' + str(i+1) + '_out'], self.outputs['B' + str(i+1) + '_cache'] = batchnorm_forward(self.outputs[current_affine_output], self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)], self.bn_params[i])
                    self.outputs[current_ReLU_output], self.outputs[current_ReLU_cache] = relu_forward(self.outputs['B' + str(i+1) + '_out'])
                else:
                    self.outputs[current_ReLU_output], self.outputs[current_ReLU_cache] = relu_forward(self.outputs[current_affine_output])
                    
                if self.use_dropout:
                    self.outputs[current_ReLU_output], self.outputs['D' + str(i+1) + '_cache'] = dropout_forward(self.outputs[current_ReLU_output], self.dropout_param)
                
        # Last layer
        self.outputs['L' + str(self.num_layers) + '_out'], self.outputs['L' + str(self.num_layers) + '_cache'] = affine_forward(self.outputs['A' + str(self.num_layers-1) + '_out'], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        
        scores = self.outputs['L' + str(self.num_layers) + '_out']
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        
        loss, dsoft = softmax_loss(scores, y) # Calculate the softmax grads and loss
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W' + str(self.num_layers)]))) # Add regularization (this continues in each layer)
        
        daffine, dw_previous, db_previous = affine_backward(dsoft, self.outputs['L' + str(self.num_layers) + '_cache'])
        grads['W' + str(self.num_layers)] = dw_previous + self.reg*self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db_previous # Update gradients and add regularization to the weights grad
        
        i = self.num_layers-1 # Initialize the index for the while loop which iterates downwards to simulate backprop
        while i > 0:
            
            if self.use_dropout:
                # Dropout backwards pass
                daffine = dropout_backward(daffine, self.outputs['D' + str(i) + '_cache'])
            
            # ReLU layer backwards pass
            drelu = relu_backward(daffine, self.outputs['A' + str(i) + '_cache']) 
            
            if self.normalization == 'batchnorm':
                
                dbatch_norm, dgamma, dbeta = batchnorm_backward_alt(drelu, self.outputs['B' + str(i) + '_cache'])
                daffine, dw_previous, db_previous = affine_backward(dbatch_norm, self.outputs['L' + str(i) + '_cache'])
                
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
                
            else:
                # Affine layer backwards pass
                daffine, dw_previous, db_previous = affine_backward(drelu, self.outputs['L' + str(i) + '_cache'])
                
                
            grads['W' + str(i)] = dw_previous + self.reg*self.params['W' + str(i)]
            grads['b' + str(i)] = db_previous
            
            loss += 0.5*self.reg*(np.sum(np.square(self.params['W' + str(i)]))) # Update loss again
        
            i -= 1
        
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
