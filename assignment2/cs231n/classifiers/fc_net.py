from builtins import range
from builtins import object
import numpy as np
from enum import Enum
from cs231n.layers import *
from cs231n.layer_utils import *

class Location(Enum):
    HIDDEN_START = 1
    HIDDEN_MIDDLE = 2
    HIDDEN_END = 3
    
class Layer(object):
    def __init__(self,Location,w,b,name):
        self.Location = Location
        self.name = name
        self.W = w
        self.B = b
        self.cache_out = None
    def get_regs(self,reg):
        return 0.5*reg * np.sum(self.W*self.W)

    def compute_back(self,d_prodact):
        if(self.Location == Location.HIDDEN_END):
            da, dw, db = affine_backward(d_prodact, self.cache_out)
        elif(self.Location == Location.HIDDEN_MIDDLE):
            da, dw, db = affine_relu_backward(d_prodact, self.cache_out)
        else:
            return
       
        return da, dw, db 
    
    def compute_foward(self,x):
        if(self.Location == Location.HIDDEN_END):
            out, cache_out = affine_forward(x, self.W, self.B)
        elif(self.Location == Location.HIDDEN_MIDDLE):
            out, cache_out = affine_relu_forward(x, self.W, self.B)
        else:
            print("this is the start point")
            return
        
        self.cache_out = cache_out
        return out
    
    def astype(self,data_type):
        self.W =  self.W.astype(data_type)
        self.B =  self.B.astype(data_type)

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
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.D = input_dim
        self.M = hidden_dim
        self.C = num_classes
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        pass
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
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        X = X.reshape(X.shape[0], self.D)
        a, a_cache = affine_relu_forward(X,W1,b1)
        scores, cache_scores  = affine_forward(a,W2,b2)
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        
        data_loss ,dscore = softmax_loss(scores,y)
        reg_loss = self.reg*np.sum(W1*W1) + self.reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        
        da, dW2, db2 = affine_backward(dscore,cache_scores)
        #adding w2 regularization derivative
        dW2 += self.reg * W2
        dx1, dW1, db1 = affine_relu_backward(da,a_cache)
        #adding w1 regularization derivative
        dW1 += self.reg * W1
        #set gradient
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
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
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.layers = {}
        self.L = len(hidden_dims) + 1
        self.N = input_dim
        self.C = num_classes
        
        dims = [self.N] + hidden_dims
        for i in range(len(dims)-1):
            w_middle =  weight_scale * np.random.randn(dims[i], dims[i + 1])
            b_middle = np.zeros(dims[i + 1])
            self.params['W' + str(i + 1)]=w_middle
            self.params['b' + str(i + 1)]=b_middle
            self.layers['layer' + str(i+1)] = Layer(Location.HIDDEN_MIDDLE,w_middle,b_middle,str(i+1))
            
        w_end =  weight_scale * np.random.randn(dims[-1], self.C)
        b_end = np.zeros(self.C)
        self.layers['layer' + str(len(dims))] =Layer(Location.HIDDEN_END,w_end,b_end,str(len(dims))) 
        self.params['W' + str(len(dims))]=w_end
        self.params['b' + str(len(dims))]=b_end
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        pass
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
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            v.astype(dtype)
    
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
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
    
        x_in = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        for i in range(self.L):
            x_in =  self.layers['layer'+str(i + 1)].compute_foward(x_in)
        scores = x_in


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        data_loss ,dscores = softmax_loss(scores,y)
        reg_loss = 0
        
        for i in range(self.L):
            reg_loss+=self.layers['layer'+str(i + 1)].get_regs(self.reg) 
        loss = data_loss + reg_loss
        hidden = {}
        d_out = dscores
        hidden['da' + str(self.L)] = dscores
        for i in range(self.L)[::-1]:
            idx = i + 1
            da = hidden['da' + str(idx)]
            layer = self.layers['layer'+str(i + 1)]
            da, dw, db = layer.compute_back(da)
            hidden['da' + str(idx - 1)] = da
            hidden['dW' + str(idx)] = dw
            hidden['db' + str(idx)] = db
 
            
        for i in range(self.L):
            idx = i + 1
            layer = self.layers['layer'+str(idx)]
            grads['W'+str(idx)] =  hidden['dW' + str(idx)] + self.reg * self.params['W'+str(idx)]
            grads['b'+str(idx)] =  hidden['db' + str(idx)] + self.reg * self.params['b'+str(idx)]
        return loss, grads
