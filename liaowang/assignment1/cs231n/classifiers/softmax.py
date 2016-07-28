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
  num_train=X.shape[0]
  num_class=W.shape[1]
  for i in range(num_train):
	P=X[i].dot(W)
	P-=np.max(P)
	los=-P[y[i]]+np.log(np.sum(np.exp(P)))
	loss+=los
	for j in range(num_class):
		if j!=y[i]:
			dW[:,j]+=(np.exp(P[j])/np.sum(np.exp(P)))*X[i]
		else:
			dW[:,j]+=(np.exp(P[j])/np.sum(np.exp(P))-1)*X[i]
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg* W 
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
  num_class=W.shape[1]
  P=X.dot(W)
  P_max=P.max(axis=1).reshape(-1,1)
  P=P-P_max
  P_yi=P[range(num_train),y].reshape(-1,1)
  los=-P_yi+np.log(np.sum(np.exp(P),axis=1)).reshape(-1,1)
  loss=np.sum(los)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  
  softmax_output=np.exp(P)/np.sum(np.exp(P),axis=1).reshape(-1,1)
  softmax_output[range(num_train),y]-=1
  dW=(X.T).dot(softmax_output)
  dW = dW/num_train + reg* W 
  
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

