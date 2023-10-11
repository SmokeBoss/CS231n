import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
    
  for i in range(num_train): # Loop through all the training data
    
    scores = X[i].dot(W) # perform the calculation of the expected scores for training point i
    correct_class_score = scores[y[i]] # extract the correct scores for training point i
    
    for j in range(num_classes): 
      if j == y[i]: # This is to ensure that we do not add the error when j = i (i.e target class)
        continue 
        
      margin = scores[j] - correct_class_score + 1 # note delta = 1, also note that this is still a vector
      if margin > 0:
        loss += margin # summation for all margins
        dW[:,j] += X[i,:]
        dW[:, y[i]] -= X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # Note for dW[:,j] += X[i,:] This is essentially saying that for each classes gradient, what we are doing is simply 
  # adding on the pixel data of an image for an incorrect classification. As a result, the weights will be more likely
  # to move the other way.
  # I still dont fully understand why we have the y[i] case but from what ive seen its important as it is in all the
  # gradients/losses. Therefore, when we perform differentiation, without the partial differentiation involving y[i]
  # ur fucked.


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  scores = np.dot(X, W) # Simple dot product to compute all the scores
  actual_scores = scores[np.arange(scores.shape[0]), y] # extracting the scores of the correct clases
  margins = np.maximum(0, scores - np.matrix(actual_scores).T + 1) # calculating the margin ALSO NOTE that broadcasting is involved here for actual scores and 1
  margins[np.arange(X.shape[0]), y] = 0 # setting the scores of the correct margins to 0
  loss = np.mean(np.sum(margins, axis = 1)) # Sum and calculate the mean over all the training examples
  loss += 0.5 * reg * np.sum(W * W) # Add the regularization term
  

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  binary = margins
  binary[margins > 0] = 1 # creating the 'mask' that we use to sum our gradients
  row_values = np.sum(binary, axis = 1) # this is important for the gradient of the correct class
  binary[np.arange(X.shape[0]), y] = -row_values.T # insert the correct values in. Python being able to do this is quite interesting
  dW = np.dot(X.T, binary)
  
  # average gradient
  dW /= X.shape[0]
    
  # apply regularization  
  dW += reg*W
    
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
