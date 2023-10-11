import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores_norm = np.zeros((num_train, num_classes))
  

  # Remember for softmax we want the distributed probability. This means that we exponentiate all our scores over an i and then divide
  # by the sum of those exponentiated scores. The loss function is going to be -log(probability of the true class)
  scores = np.dot(X, W) # compute the respective scores for each class for each example
  scores -= np.max(scores) # normalization so we dont run out of memory                       
  scores_exp = np.exp(scores) # compute probablilities

  scores_exp_sum = np.sum(scores_exp, axis = 1) # sum up all the probability scores for the denominator
  
  for i in range(num_train): 
      scores_norm[i,:] = scores_exp[i,:] / scores_exp_sum[i]  # normalize all the scores and store them    
      loss += -np.log(scores_norm[i, y[i]]) # this gets the true classes score, inverse logs it, and adds it to loss
      
      for k in range(num_classes):
          p_k = scores_norm[i,k] # extract the score of class k
          dW[:,k] += (p_k - (y[i] == k))*X[i,:] # Gradeint update that affects a classes weight for all examples
      
                        
                         
  loss /= num_train # this averages the loss over all examples 
  loss += 0.5 * reg * np.sum(W*W) # regularization     
                         
  dW /= num_train # average the gradeint updates
  dW += reg*W                       

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  num_train = X.shape[0]
    
  scores = np.dot(X, W) # compute the respective scores for each class for each example
  scores -= np.max(scores) # normalization so we dont run out of memory                       
  scores_exp = np.exp(scores) # compute probablilities  
  
  scores_exp_sum = np.sum(scores_exp, axis = 1)[..., np.newaxis] # sum up all the probability scores for the denominator, newaxis is for broadcasting purposes

  scores_norm = scores_exp/scores_exp_sum  # normalize all the scores and store them
    
  loss = np.sum(-np.log(scores_norm[np.arange(0, num_train), y])) # this gets the true classes score, inverse logs it, and adds it to loss

  
  # Grad calcs which are spesh
  ones = np.zeros_like(scores_norm) # create a array which corresponds to all the probabilities so we can decide for which y = k
  ones[np.arange(0, num_train), y] = 1 # assign ones for when this is the case
  dW = np.dot(X.T, scores_norm - ones) # backprop of the scores and X to calculate dW

  loss /= num_train # this averages the loss over all examples 
  loss += 0.5 * reg * np.sum(W*W) # regularization     
                         
  dW /= num_train # average the gradeint updates
  dW += reg*W  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

