import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores_i = X[i].dot(W) # 1 * C
    scores_i -= max(scores_i)
    loss_i = -scores_i[y[i]] + np.log(sum(np.exp(scores_i)))
    loss += loss_i
    for j in xrange(num_classes):
      softmax_output = np.exp(scores_i[j])/sum(np.exp(scores_i))
      if j == y[i]:
        dW[:, j] += (-1 + softmax_output) * X[i]
      else:
        dW[:, j] += softmax_output * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW = dW /num_train + reg * W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores = scores - (scores.max(axis=1)).reshape(-1,1)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1)
  loss = -correct_class_scores + np.log(np.exp(scores).sum(axis=1).reshape(-1,1))
  loss = sum(loss)/num_train
  loss += 0.5 * reg * np.sum(W*W)

  softmax_output = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1,1)
  softmax_output[range(num_train), list(y)] += -1
  dW = (X.T).dot(softmax_output)
  dW = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

