import numpy as np
from .nn.linear import *
from .nn.activation import *
from mytorch.functional import *
from mytorch.autograd_engine import *

class RNNCell(object):
    """RNN Cell class."""
    def __init__(self, input_size, hidden_size, autograd_engine, act_fn=Tanh):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.autograd_engine = autograd_engine

        # Activation function for cell
        self.activation = act_fn(self.autograd_engine)

        # initialize two linear layers
        self.ih = Linear(self.input_size, self.hidden_size, self.autograd_engine)
        self.hh = Linear(self.hidden_size, self.hidden_size, self.autograd_engine)

        # Gradients
        self.zero_grad()
        
    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.ih.init_weights(W_ih,  b_ih)
        self.hh.init_weights(W_hh, b_hh)

    def zero_grad(self):
        self.ih.zero_grad() 
        self.hh.zero_grad() 
    
    def __call__(self, x, h_prev_t, scale_h=None):
        return self.forward(x, h_prev_t, scale_h)

    def forward(self, x, h_prev_t, scale_h=None):
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
       
        i1  = self.ih(x)
        i2_ = self.hh(h_prev_t)

        # NOTE: Optional scale factor included to make this class 
        # available for use in GRUCell while computing self.n
        if scale_h is not None:
            i2  = scale_h * i2_
            self.autograd_engine.add_operation(inputs=[scale_h, i2_], output=i2,
                                               gradients_to_update=[None, None],
                                               backward_operation=mul_backward)
        else:
            i2 = i2_.copy()
            self.autograd_engine.add_operation(inputs=[i2_], output=i2,
                                               gradients_to_update=[None],
                                               backward_operation=identity_backward)


        i = i1 + i2
        self.autograd_engine.add_operation(inputs=[i1, i2], output=i,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        h_t = self.activation.forward(i)
        return h_t
