from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    pass

    N_dim = x.shape[0]
    x_shapes = x.shape
    
    x = x.reshape([N_dim, -1]) # Reshape and preserve the rows. The column info is inferred from the array information
    
    out = np.dot(x, w) + b # Matrix multiply and add bias
    x.shape = x_shapes

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass

    
    N_dim = x.shape[0]


    new_x = np.copy(x)
    new_x = new_x.reshape(N_dim, -1)
    
    # The derivative of dx is the sum of the derivative of the error with respect to the next layers pre-activations (dout) element multiplied with the weights and summed over all the connected neurons from x to the next layer. i.e dC/dx = sum_over_all_neurons(dC/dz * w(from z to x)). When we transpose W, and do a dot product between dout and w.T we perform this calculation in 1 matrix operation over the entire connected network. http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    
    dx = np.dot(dout, w.T) 
    
    dx.shape = x.shape # reshape the array to the origional x shape
    
    # Similarly with the previous case but we can easily figure this one out if we combine the preious ideas with the cs231n paradigm of backpropogation. i.e dC/dW = dC/dZ * dZ/dW where Z is the pre-activation values. Note that since wx + b = z dZ/dW = x. Also, we must sum this over all the previous layers neurons that feed into the current one. In this case, the fact that we already have x (the inputs which we would typically sum over arranged as 1 matrix, we can can easily do this. In the previous case, each row of weights corresponded to a certain neuron from the previous layer. Now, each column of x corresponds to 
    
    dw = np.dot(new_x.T, dout) 
    
    
    db = np.sum(dout, axis = 0) # db has no derivative and is just dout. Note the sum which ensures the dimensions match
  
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    pass

    out = np.maximum(0, x) # Broadcast 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    pass
    

    dx = np.zeros(x.shape) # initialize dx to be the correct shape
    dx[x > 0] = 1 # basically, every grad is 0 for x <= 0 and 1 for x > 0
    dx = dx*dout


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        pass

        
        # Calculate the mean and variance
        mu = np.mean(x, axis = 0) # axis 0 works down the columns (or the dimensions)
        variance = np.mean((x-mu)**2, axis = 0) # Broadcast mu 
        
        x_norm = (x - mu)/np.sqrt(variance + eps) # Normalization
        
        out = gamma*x_norm + beta # Output
        
        # Calculate the running means for test time behaviour
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * variance
        
        # Needed for cache: variance, x, gamma, mu, eps, m(number of training examples), x_norm
        cache = {
            'variance': variance,
            'x': x,
            'gamma': gamma,
            'mu': mu,
            'eps': eps,
            'x_norm': x_norm,
            'm': N,
        }
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        pass
        
        x_norm = ( x - running_mean) / np.sqrt(running_var + eps) # normalization with running means
        out = gamma*x_norm + beta # output
        
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    pass
    # dout is dy
    # Needed for cache: variance, x, gamma, mu, eps, m(number of training examples), x_norm
    
    variance = cache.get('variance')
    x = cache.get('x')
    gamma = cache.get('gamma')
    mu = cache.get('mu')
    eps = cache.get('eps')
    m = cache.get('m')
    x_norm = cache.get('x_norm')
    

    
    # Using the gradient maps and by working backwards we can start to unravel the gradient
    
    # Summation Gate
    dbeta = np.sum(1*dout, axis = 0) # Note that each feature has its own gamma and beta (matching their mu and sigma^2)
    d1 = 1*dout
    
    # Multiplication Gate
    dgamma = np.sum(x_norm*d1, axis = 0)
    d2 = d1 * gamma 
    
    # Multiplication Gate
    dxmu = d2*(1/np.sqrt(variance+eps)) # the ./ casts to float. Without it we would have integer division 
    d3 = np.sum(d2*(x - mu), axis = 0) 
    
    # Inversion Gate
    d4 = d3*(-1/(np.sqrt(variance + eps))**2) 
    
    # Sq Root Gate
    d5 = d4*(0.5 * (1/np.sqrt(variance + eps)))#
    
    # Summation Gate
    d6 = d5 # d6 is dvariance
    
    # Mean Gate
    # This gate is somewhat special, how we deal with it is by evenly distributing the upstream gradient over all of the rows.
    d7 = (1/m) * d6 * np.ones_like(dout) 
    
    # Sq Gate
    d8 = d7 * (x-mu)*2 
    
    # Subtraction Gate 
    # This gate is also interesting since we have 2 upstream gradients which means, in this case, we need to add them
    d0x = 1*(dxmu + d8) 
    d9 = -1 * np.sum(dxmu + d8, axis = 0) 
    
    # Mean Gate
    d10 = (1/m) * d9 * np.ones_like(dout)
    
    # Initial dx, which is a summation of the 2 upstream grads
    dx = d0x + d10

    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    
    # Extract cache values
    variance = cache.get('variance')
    x = cache.get('x')
    gamma = cache.get('gamma')
    mu = cache.get('mu')
    eps = cache.get('eps')
    m = cache.get('m')
    x_norm = cache.get('x_norm')
    

    # Perform gradient calcs
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (variance + eps)**(-1. / 2.) * dout, axis=0)
    dx = (1. / m) * gamma * (variance + eps)**(-1. / 2.) * (m * dout - np.sum(dout, axis=0) - (x - mu) * (variance + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    pass
    # Layer norm is simply just normalizing the input over the features. I.e you normalize each training example. With that being the case, you can simply just sum over the columns instead of over the rows as we did previously.
    
    
    #x = x.T # Transpose x to use the origional batch norm code
    
    N, D = np.shape(x) # Note here that we use the new shape, N now notates the dimensions instead of the data
    
    mu = np.mean(x, axis = 1)[:, None] # We calculate a mean over the training examples
    variance = np.mean((x-mu)**2, axis = 1)[:, None] # Broadcast mu and calculate mean over the training examples. 
    
    x_norm = (x - mu)/np.sqrt(variance + eps) # Normalization
    
    
    #x_norm = x_norm.T # Transform for the output calculation
    out = gamma*x_norm + beta # Output
    
    # Needed for cache: variance, x, gamma, mu, eps, m(number of training examples), x_norm
    cache = {
        'variance': variance,
        'x': x,
        'gamma': gamma,
        'mu': mu,
        'eps': eps,
        'x_norm': x_norm,
        'm': D,
    }    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass
    
        # Extract cache values
    variance = cache.get('variance')
    x = cache.get('x')
    gamma = cache.get('gamma')
    mu = cache.get('mu')
    eps = cache.get('eps')
    m = cache.get('m')
    x_norm = cache.get('x_norm')
    
    
    
    # Summation Gate
    dbeta = np.sum(1*dout, axis = 0) # Note that each feature has its own gamma and beta (matching their mu and sigma^2)
    d1 = 1*dout
    
    # Multiplication Gate
    dgamma = np.sum(x_norm*d1, axis = 0)
    d2 = d1 * gamma 
    
    # Multiplication Gate
    dxmu = d2*(1/np.sqrt(variance+eps)) # the ./ casts to float. Without it we would have integer division 
    d3 = np.sum(d2*(x - mu), axis = 1)[:, None] 
    
    # Inversion Gate
    d4 = d3*(-1/(np.sqrt(variance + eps))**2) 
    
    # Sq Root Gate
    d5 = d4*(0.5 * (1/np.sqrt(variance + eps)))
    
    # Summation Gate
    d6 = d5 # d6 is dvariance
    
    # Mean Gate
    # This gate is somewhat special, how we deal with it is by evenly distributing the upstream gradient over all of the columns.
    d7 = (1/m) * d6 * np.ones_like(dout) 
    
    # Sq Gate
    d8 = d7 * (x-mu)*2 
    
    # Subtraction Gate 
    # This gate is also interesting since we have 2 upstream gradients which means, in this case, we need to add them
    d0x = 1*(dxmu + d8) 
    d9 = -1 * np.sum(dxmu + d8, axis = 1)[:, None] 
    
    # Mean Gate
    d10 = (1/m) * d9 * np.ones_like(dout)
    
    # Initial dx, which is a summation of the 2 upstream grads
    dx = d0x + d10
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        pass
        
        mask = (np.random.rand(*x.shape) < p) / p # Inverted dropout. The /p is there to reduce test time
        out = x*mask # The actual dropout
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        pass
        
        out = np.maximum(0, x)
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        
        dx = dout*mask 
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W) i.e Num_Examples, RGB Channels, Height, Width
    - w: Filter weights of shape (F, C, HH, WW) i.e Number of Filters, Depth of Filter, Height of Filter, Width of Filter
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass
    
    f_num, f_layers, f_heights, f_widths = np.shape(w)
    x_num, x_layers, x_heights, x_widths = np.shape(x)
    x_new = np.copy(x)
    
    # With a stride of 1 padding is determined by p = (f-1)/2
    p = conv_param.get('pad')
    stride = conv_param.get('stride')
    
    pad_tuples = ((0,0), (0,0), (p,p), (p,p)) # Create a tuple indicating how much we want to pad on either side. In this case, we want 1 row above and below the data and 1 column above and below the data.
    x_new = np.pad(x_new, pad_tuples, 'constant')
    
    new_height = np.int((x_heights + 2*p - f_heights)/stride + 1)
    new_width = np.int((x_widths + 2*p - f_widths)/stride + 1)
    
    out = np.zeros([x_num, f_num, new_height, new_width]) # initialize the output layer
    
    for N in range(x_num): # Note that the python array goes from 0 to x_num-1. This is the same for the rest of the loops
        for i in range(f_num):
            
            for j in range(new_height):
                for k in range(new_width):
                    
                    # Convolution layer. Note the stride*j:j+f_heights which ensures we are working over the correct receptive field. Also, np.dot will not work in this case since we are working with 3d arrays. Hence the sum is needed. Note that for the stride part we dont need to minus f_widths because it isnt needed. The demo page is useful for this visualization
                    out[N, i, j, k] = np.sum(x_new[N, :, (stride*j):(stride*j+f_heights), (stride*k):(stride*k+f_widths)]*(w[i, :, :, :]))
                    out[N, i, j, k] += b[i] # Add the bias to the layer
            

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    
    x, w, b, conv_param = cache
    
    p = conv_param.get('pad')
    stride = conv_param.get('stride')
    
    x_new = np.copy(x)
    # Re-pad to develop the correct receptive field
    pad_tuples = ((0,0), (0,0), (p,p), (p,p))
    x_new = np.pad(x_new, pad_tuples, 'constant') # Note that x must be the same as the padded x we used
    
    dw = np.zeros(np.shape(w))
    dx_temp = np.zeros(np.shape(x_new)) # Dx must also be padded in the mean time
    db = np.zeros(np.shape(b))
    
    f_num, f_layers, f_heights, f_widths = np.shape(w)
    x_num, x_layers, x_heights, x_widths = np.shape(x)
    x_num, f_num, out_height, out_width = np.shape(dout)

    
    # Ok, the derivative of weight is basically a dot product between dout and x over the receptive field. 
    for i in range(f_num): # This was swapped for an easier calculation of db
        db[i] += np.sum(dout[:, i, :, :]) # db is just a sum of dout by layer F
        for N in range(x_num):
            for j in range(out_height):
                for k in range(out_width):
                    
                    # Note that this is basically the corresponding dout element * receptive field. Which is then summed
                    dw[i, :, :, :] += dout[N, i, j, k]*x_new[N, :, (stride*j):(stride*j+f_heights), (stride*k):(stride*k+f_widths)]
                    
                    # dx is the same when you think about the receptive fields
                    dx_temp[N, :, (stride*j):(stride*j+f_heights), (stride*k):(stride*k+f_widths)] += dout[N, i, j, k]*w[i, :, :, :]
               
    # Remoe the padding on dx                        
    dx = dx_temp[:, :, p:x_heights+p, p:x_widths+p]                
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pass
    
    N, C, Height, Width = x.shape
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    
    H_out = int((Height - (pool_height)) / stride + 1) # calculate correct output dimensions
    W_out = int((Width - (pool_width)) / stride + 1) # Casting to int is neccesary for np.zeros
    
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(N):
        for j in range(C):
            for k in range(H_out):
                for l in range (W_out): # Find the max over the correct receptive field per output place
                    out[i, j, k, l] = np.max(x[i, j, stride*k:stride*k+pool_height, stride*l:stride*l+pool_width])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    pass

    x, pool_param = cache
    
    N, C, Height, Width = x.shape
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    
    H_out = int((Height - (pool_height)) / stride + 1) # calculate correct output dimensions
    W_out = int((Width - (pool_width)) / stride + 1) # Casting to int is neccesary for np.zeros
    
    out = np.zeros((N, C, H_out, W_out))
    
    maximum_shape = (1, 1, pool_height, pool_width)
    
    dx = np.zeros(x.shape)
    
    for i in range(N):
        for j in range(C):
            for k in range(H_out):
                for l in range (W_out): # Find the max over the correct receptive field per output place
                    #out[i, j, k, l] = np.max(x[i, j, stride*k:stride*k+pool_height, stride*l:stride*l+pool_width])
                    
                    # The reshape turns a 2d array into a 1d array that stretches along 1 row. i.e multiple rows added to 1
                    # We work in this dimension since numpy will arrange the data along the row with spillover going into the next row
                    new_array = np.reshape(x[i, j, stride*k:stride*k+pool_height, stride*l:stride*l+pool_width], (1, -1))
                    #print(x[i, j, stride*k:stride*k+pool_height, stride*l:stride*l+pool_width])
                    
                    maximum = np.argmax(new_array)
                    height_index = int(maximum / pool_width) # Calculate the width and height indices according to how x was rearranged
                    width_index = maximum % pool_width
                    #print('width: ' + str(width_index))
                    #print('height: ' + str(height_index))
                    
                    # The correct index for dx is updated with the upstream gradient
                    dx[i, j, stride*k+height_index, stride*l+width_index] = 1*dout[i, j, k, l]
                    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass


    # To use the origional batchnorm we need to have the dimension we are normalizing over on the last dimension (columns) and a 2d matrix
    # Note that the transpose list puts the corresponding axis number (0-n from left to right respectively) into that place.
    
    x_t = np.transpose(x, (0, 2, 3, 1)) # Here we've just shifted the layers to the last axis 
    x_new = np.reshape(x_t, (-1, x.shape[1])) # Reshape into a 2 dimensional matrix so that we can use the original code
    
    x_t_norm, cache = batchnorm_forward(x_new, gamma, beta, bn_param) # Call the origional code
    
    out = np.transpose(np.reshape(x_t_norm, (x_t.shape)), (0, 3, 1, 2)) # Undo the transform
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    
    # Reshape like we did with x in the forward pass
    dout_t = np.transpose(dout, (0, 2, 3, 1))  
    dout_new = np.reshape(dout_t, (-1, dout.shape[1])) 
    
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache) # Call the backwards pass function. Note that dgamma and dbeta dont need to be reshaped
    
    dx = np.transpose(np.reshape(dx, (dout_t.shape)), (0, 3, 1, 2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    
    """
    
    N, C, Height, Width = x.shape
    
    # Reshape so that we are able to correctly calculate the mean and variance. Note that a new axis was introduced (C//G) which holds the previous data but is now introduced in a new format with g 'bins' // indicates a floor division
    x_group = np.reshape(x, (N, G, C//G, Height, Width)) 
    
    mean = np.mean(x_group, axis = (2, 3, 4), keepdims = True) # Calculate the mean over the matrix axis provided
    variance = np.var(x_group, axis = (2, 3, 4), keepdims = True) 
    
    x_norm = (x_group - mean) / np.sqrt(variance + eps)
    
    x_norm = np.reshape(x_norm, (N, C, Height, Width))
    
    out = x_norm*gamma + beta
    
    cache = {
        'variance': variance,
        'x': x,
        'gamma': gamma,
        'mu': mean,
        'eps': eps,
        'x_norm': x_norm,
        'm': (C//G)*Height*Width,
        'G': G
    }    
    """
    N, C, Height, Width = x.shape
    
    # Noting that the paper recommends this transform: (N, G, C//G, Height, Width), we can repurpose for the code we have written
    x_new = np.reshape(x, (N*G, C // G * Height * Width)).T # Reshape and transpose to get the rows and columns into a format where we can use batchnorm
    
    # Calculate the mean and variance 
    mean = np.mean(x_new, axis = 0)
    variance = np.var(x_new, axis = 0)
    
    x_norm = (x_new - mean) / np.sqrt(variance + eps)
    x_norm = np.reshape(x_norm.T, (N, C, Height, Width)) # Reverse the transformation
    
    out = gamma*x_norm + beta
    
    
    cache = {
        'variance': variance,
        'x': x,
        'gamma': gamma,
        'mu': mean,
        'eps': eps,
        'x_norm': x_norm,
        'm': (C//G)*Height*Width,
        'G': G
    }    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass

    N, C, Height, Width = dout.shape
    
    # dout is dy
    # Needed for cache: variance, x, gamma, mu, eps, m(number of training examples), x_norm
    
    variance = cache.get('variance')
    x = cache.get('x')
    gamma = cache.get('gamma')
    mu = cache.get('mu')
    eps = cache.get('eps')
    m = cache.get('m')
    x_norm = cache.get('x_norm')
    G = cache.get('G')
    
    x_new = np.reshape(x, (N*G, C // G * Height * Width)).T 
    
    dbeta = np.sum(1*dout, axis = (0, 2, 3), keepdims = True) # Note that we now sum over different axis 
    d1 = 1*dout
    
    dgamma = np.sum(x_norm*d1, axis = (0, 2, 3), keepdims = True)
    d2 = d1 * gamma
    
    # Now we do the transformations for the grad and x_norm
    d2_new = np.reshape(d2, (N*G, C // G * Height * Width)).T 
    x_norm_new = np.reshape(x_norm, (N*G, C // G * Height * Width)).T 
    
    
    # Copied relevant batchnorm code
    
    # Multiplication Gate
    dxmu = d2_new*(1/np.sqrt(variance+eps)) # the ./ casts to float. Without it we would have integer division 
    d3 = np.sum(d2_new*(x_new - mu), axis = 0) 
    
    # Inversion Gate
    d4 = d3*(-1/(np.sqrt(variance + eps))**2) 
    
    # Sq Root Gate
    d5 = d4*(0.5 * (1/np.sqrt(variance + eps)))#
    
    # Summation Gate
    d6 = d5 # d6 is dvariance
    
    # Mean Gate
    # This gate is somewhat special, how we deal with it is by evenly distributing the upstream gradient over all of the rows.
    d7 = (1/m) * d6 * np.ones_like(x_new) 
    
    # Sq Gate
    d8 = d7 * (x_new-mu)*2 
    
    # Subtraction Gate 
    # This gate is also interesting since we have 2 upstream gradients which means, in this case, we need to add them
    d0x = 1*(dxmu + d8) 
    d9 = -1 * np.sum(dxmu + d8, axis = 0) 
    
    # Mean Gate
    d10 = (1/m) * d9 * np.ones_like(x_new)
    
    # Initial dx, which is a summation of the 2 upstream grads
    dx = d0x + d10
    
    
    dx = np.reshape(dx.T, (N, C, Height, Width)) # Transform dx back to the correct shape

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0] # retrieve number of samples
    correct_class_scores = x[np.arange(N), y] # retrive the class scores from x matrix
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0) # margin calc with a safety of 1
    margins[np.arange(N), y] = 0 # ensure that the margins on the correct classes are 0
    loss = np.sum(margins) / N # average the loss per class
    num_pos = np.sum(margins > 0, axis=1) # number of incorrect assignments per example?
    dx = np.zeros_like(x)
    dx[margins > 0] = 1 
    dx[np.arange(N), y] -= num_pos # update grad for correct class weights
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True) # Numeric stability fixes
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True) 
    log_probs = shifted_logits - np.log(Z) 
    probs = np.exp(log_probs) # Probabilities from softmax formula
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N # loss from softmax formula
    dx = probs.copy()
    dx[np.arange(N), y] -= 1 # Gradient calcs from formula
    dx /= N
    return loss, dx
