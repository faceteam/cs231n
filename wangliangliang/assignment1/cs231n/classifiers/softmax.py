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
  y_pre=np.zeros([X.shape[0],W.shape[1]])
  for xi in range(X.shape[0]):
  	temp=0
  	for wj in range(W.shape[1]):
  		for xj in range(X.shape[1]):
  			wi=xj
  			y_pre[xi][wj]+=X[xi][xj]*W[wi][wj]
  		y_pre[xi][wj]=np.exp(y_pre[xi][wj])
  		temp+=y_pre[xi][wj]
  	for yj in range(y_pre.shape[1]):
  		y_pre[xi][yj]=y_pre[xi][yj]/temp
  		if y[xi]==yj:
  			for dwi in range(W.shape[0]):
  				dW[dwi][yj]+=(y_pre[xi][yj]-1)*X[xi][dwi]
  		else:
  			for dwi in range(W.shape[0]):
  				dW[dwi][yj]+=y_pre[xi][yj]*X[xi][dwi]
  	loss+=-np.log(y_pre[xi][y[xi]])
  re=0
  dW=dW/X.shape[0]
  for i in range(W.shape[0]):
  	for j in range(W.shape[1]):
  		re+=W[i][j]**2
  re=re*0.5

  loss=loss/X.shape[0]+reg*re
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
  y_pre=np.exp(np.dot(X,W))
  sum_all=np.sum(y_pre,axis=1).reshape(y_pre.shape[0],-1)
  y_pre=y_pre/sum_all
  y_parse=np.zeros_like(y_pre)
  y_bp=y_pre.copy()
  index=0
  for i in y:
  	y_parse[index][i]=1
  	y_bp[index][i]=y_pre[index][i]-1
  	index+=1
  loss=np.mean(-np.log(np.sum(y_pre*y_parse,axis=1)))+0.5*reg*np.sum(np.square(W))
  for i in range(X.shape[0]):
    dW+=y_bp[i]*(X[i].reshape(X.shape[1],-1))
    dW=dW/X.shape[0]
  dW+=reg*W





  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

