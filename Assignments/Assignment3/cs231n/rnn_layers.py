from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    pass
    
    next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    pass
    
    x, prev_h, Wx, Wh, b, next_h = cache

    
    
    # the gradient of tanh(z) is 1-tanh(z)^2, and next_h is tanh(z)
    dtan = dnext_h * (1 - next_h ** 2)
    
    
    
    # Classic BackProp
    db = np.sum(dtan, axis = 0)
    dx = np.dot(dtan, Wx.T)
    dWx = np.dot(x.T, dtan)
    dprev_h = np.dot(dtan, Wh.T)
    dWh = np.dot(prev_h.T, dtan)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    pass
    
    N, T, D = x.shape
    N, H = h0.shape
    # Initialization of tuple and prevh for loop
    cache = dict()
    h = np.zeros((N, T, H))
    prev_h = h0
    
    for i in range(T):
        h[:, i, :], cache[i] = rnn_step_forward(x[:, i, :], prev_h, Wx, Wh, b) # Continously call step_forward function to populate tuple
        prev_h = h[:, i, :]
    
        
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    pass
    
    N, T, H = dh.shape
    
    # Extract the cache of the last node
    x, prev_h, Wx, Wh, b, next_h = cache[T-1]
    
    D, H = Wx.shape
    
    #Initialize the values
    dx = np.zeros((N, T, D))
    dh0 = np.zeros_like(prev_h)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    
    # We're resetting prev_h to zeros here because the only dependancy is the loss which is accounted for later
    dprev_h = np.zeros_like(prev_h) 
    
    
    for i in range(T):
        # From http://www.deeplearningbook.org/contents/rnn.html Eq. 10.20, which basically says that we need to add the gradients dh onto the dprev_h we get. This is to account for the 2 dependancies, and will (probably) help to alleviate the vanishing gradient issue. 
        dprev_h += dh[:, T-(i+1), :]
        
        # Call the step function, note that we put x directly into the gradient matrix
        dx[:, T-(i+1), :], dprev_h, dWx_step, dWh_step, db_step = rnn_step_backward(dprev_h, cache[T-(i+1)]) 
        
        # Summation of gradients due to their many dependancies
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    
    # dh0 will only depend on dprev_h
    dh0 = dprev_h
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    pass
    
    out = W[x] # This line will take all the indexes stored at every position in x and call it in W
    cache = x, W
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    pass
    
    x, W = cache
    
    dW = np.zeros_like(W)
    
    # This lines adds each point in dout (or a dimension of points) into dW according to the first 2 dimensions of x
    np.add.at(dW, x, dout) 
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    pass
    N, H = prev_h.shape
    
    # Note that we have an array not a vector. Due to the fact that we have batches of data, not individual points
    # Compute the activation array a
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    
    # Slice up the activation array for each gate
    a_i = a[:,:H]
    a_f = a[:,H:H*2]
    a_o = a[:,H*2:H*3]
    a_g = a[:,H*3:]
    
    #print(H)
    #print(a_i.shape)
    #print(a_f.shape)
    
    # Sigmoid activation function for input gate, forget gate, and ooutput gate
    i = 1. /(1 + np.exp(-1*a_i))
    f = 1. /(1 + np.exp(-1*a_f))
    o = 1. /(1 + np.exp(-1*a_o))
    # Tanh activation function for gate gate
    g = np.tanh(a_g)
    
    # Next cell state
    next_c = f * prev_c + i * g
    
    # Next block state
    next_h = o * np.tanh(next_c)
    
    cache = i, f, o, g, next_c, prev_c, prev_h, Wx, Wh, x
    # a, a_i, a_f, a_o, a_g,
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    pass
    # Retrieve cached variables
    i, f, o, g, next_c, prev_c, prev_h, Wx, Wh, x = cache
    
    d_preactivation_c = dnext_h * o
    dnext_c += d_preactivation_c * (1 - np.tanh(next_c)**2)
    
    do = dnext_h * np.tanh(next_c)
    
    df = dnext_c * prev_c
    dprev_c = dnext_c * f
    di = dnext_c * g
    dg = dnext_c * i
    
    # Tanh derivative (1 - tanh^2)
    da_g = dg * (1 - g**2)
    
    # Sigmoid derivatives (sigma * (1 - sigma))
    da_i = di * (i*(1 - i))
    da_f = df * (f*(1 - f))
    da_o = do * (o*(1 - o))
    
    # Join the arrays along the columns, as intended
    da = np.concatenate([da_i, da_f, da_o, da_g], axis = 1) 
    
    # Sum along the rows for db
    db = np.sum(da, axis = 0)
    
    dWx = np.dot(x.T, da)
    dx = np.dot(da, Wx.T)
    
    dWh = np.dot(prev_h.T, da)
    dprev_h = np.dot(da, Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    
    N, T, D = x.shape
    N, H = h0.shape
    
    # Initialization 
    cache = dict()
    h = np.zeros((N, T, H))
    prev_c = np.zeros((N, H))
    prev_h = h0
    
    for i in range(T):
        # Call the lstm step function, paying attention to the fact that we store into h first
        h[:, i, :], prev_c, cache[i] = lstm_step_forward(x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = h[:, i, :]    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass

    N, T, H = dh.shape
    
    # Retrieve the last set of cached variables
    i, f, o, g, next_c, prev_c, prev_h, Wx, Wh, x = cache[T-1]
    D, _ = Wx.shape
    
    # Initialize the gradient arrays
    dx = np.zeros((N, T, D))
    dh0 = np.zeros_like(prev_h)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((H*4,))
    
    
    dprev_c = np.zeros_like(next_c)
    dprev_h = np.zeros_like(prev_h)
    
    for i in range(T):
        # This is the same concept as with the RNN stepback function
        dprev_h += dh[:, T-(i+1), :]
        
        dx[:, T-(i+1), :], dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dprev_h, dprev_c, cache[T-(i+1)])
        
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
        
    # After the for loop has run we assign dh0 (since it only has 1 dependancy, we dont need to add on dh)
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
