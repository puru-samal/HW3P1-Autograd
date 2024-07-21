import numpy as np
from .nn.linear import *
from .nn.activation import *
from mytorch.functional import *
from mytorch.autograd_engine import *

class RNNCell(object):
    """RNN Cell class."""
    def __init__(self, input_size, hidden_size, autograd_engine):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.autograd_engine = autograd_engine

        # Activation function for
        self.activation = Tanh(self.autograd_engine)

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # initialize two linear layers
        self.ih = Linear(d, h, self.autograd_engine)
        self.hh = Linear(h, h, self.autograd_engine)

        # Weights and biases
        self.ih.W = np.random.randn(h, d)
        self.hh.W = np.random.randn(h, h)
        self.ih.b = np.random.randn(h, 1)
        self.hh.b = np.random.randn(h, 1)

        # Gradients
        self.zero_grad()
        

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.ih.W = W_ih
        self.hh.W = W_hh
        self.ih.b = b_ih
        self.hh.b = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.ih.dW = np.zeros((h, d))
        self.hh.dW_hh = np.zeros((h, h))
        self.ih.db = np.zeros(h)
        self.hh.db = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """

        i1 = self.ih(x)
        i2 = self.hh(h_prev_t)
        i = i1 + i2
        self.autograd_engine.add_operation(inputs=[i1, i2], output=i,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        h_t = self.activation.forward(i)
        return h_t
