from builtins import range
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
    D , C = W.shape
    N     = X.shape[0] 


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f = np.dot(X, W)

    f -= np.max(f, axis=1, keepdims=True)
    e_f = np.exp(f)
    sum_e_f = np.sum(e_f, axis=1, keepdims=True)
    p = e_f / sum_e_f
    L = - np.log( p[np.arange(N), y] )

    loss = np.sum(L, axis=0)

    # indicator[np.arange(N), y] = 1.0
    for i in range(N):
      for c in range(C):
        dW[:, c] += p[i, c]*X[i].T
      dW[:, y[i]] -= X[i].T

    
    loss /= N 
    dW /= N 

    loss += reg*np.sum(W*W)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D , C = W.shape
    N     = X.shape[0] 
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # f 

    f = np.dot(X, W)

    f -= np.max(f, axis=1, keepdims=True)
    e_f = np.exp(f)
    sum_e_f = np.sum(e_f, axis=1, keepdims=True)
    p = e_f / sum_e_f
    L = - np.log( p[np.arange(N), y] )

    loss = np.sum(L, axis=0)

    # p   : N x C
    # X[i]: 1 x D
    # X   : N x D
    # W   : D x C

    dp = np.zeros_like(p)
    dp = p 
    dp[range(N), y] -= 1

    dW = np.dot(X.T, dp)

    loss /= N 
    dW /= N 

    loss += reg*np.sum(W*W)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
