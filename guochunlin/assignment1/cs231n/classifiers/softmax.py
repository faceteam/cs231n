import numpy as np
from random import shuffle
from past.builtins import xrange
import math
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
  num_train=X.shape[0]
  num_classes=W.shape[1]
  for i in range(num_train):
    scores=X[i].dot(W)
    exp_scores=np.zeros(scores.shape)
    row_sum=0
    for j in range(num_classes):
      exp_scores[j]=np.exp(scores[j])
      row_sum=row_sum+exp_scores[j]
    loss=loss-np.log(exp_scores[y[i]]/row_sum)
    for k in range(num_classes):
      if k !=y[i]:
        dW[:,k] +=exp_scores[k]/row_sum * X[i]
      else:
        dW[:,y[i]] +=(exp_scores[y[i]]/row_sum-1)*X[i]
  loss=loss/num_train
  loss=loss + reg * np.sum(W*W)
  dW=dW/num_train
  dW=dW+ 2 * reg * W
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
  num_train=X.shape[0]
  # 计算loss
  scores=X.dot(W)
  shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape((num_train, 1))
  loss=-np.sum(np.log(softmax_output[range(num_train),list(y)]))#计算loss这一步要注意进行log运算与sum运算的前后区别
  loss /=num_train
  loss +=0.5 *reg*np.sum(W*W)
  dS=softmax_output.copy()
  dS[range(num_train),list(y)]+=-1 #求导公式，在对Wyi进行求导时会多一个-1
  dW=(X.T).dot(dS)
  dW=dW / num_train+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

