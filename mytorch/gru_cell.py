import numpy as np
from .nn.activation import *
from .nn.linear import *
from mytorch.rnn_cell import *
from mytorch.functional import *
from mytorch.autograd_engine import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size, autograd_engine):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.autograd_engine = autograd_engine

         # initialize three linear layers
        self.r_cell = RNNCell(self.input_size, self.hidden_size, self.autograd_engine, act_fn=Sigmoid)
        self.z_cell = RNNCell(self.input_size, self.hidden_size, self.autograd_engine, act_fn=Sigmoid)
        self.n_cell = RNNCell(self.input_size, self.hidden_size, self.autograd_engine, act_fn=Tanh)
        
        # Init Gradients
        self.zero_grad()

        # Define other variables to store forward results for backward here
        self.r = None
        self.z = None
        self.n = None

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.r_cell.init_weights(Wrx, Wrh, brx, brh)
        self.z_cell.init_weights(Wzx, Wzh, bzx, bzh)
        self.n_cell.init_weights(Wnx, Wnh, bnx, bnh)

    def zero_grad(self):
        self.r_cell.zero_grad()
        self.z_cell.zero_grad()
        self.n_cell.zero_grad()
    
    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        # NOTE: Using RNNCells to compute the transformations below:
        # NOTE: r = σ (W_ir x + b_ir + W_hr h + b_hr)
        # NOTE: z = σ (W_iz x + b_iz + W_hz h + b_hz)
        # NOTE: n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        self.r = self.r_cell(self.x, self.hidden)
        self.z = self.z_cell(self.x, self.hidden)
        self.n = self.n_cell(self.x, self.hidden, scale_h=self.r)

        # NOTE: h' = (1 - z) * n + z * h
        h1 = 1 - self.z
        self.autograd_engine.add_operation(inputs=[np.ones_like(self.z), self.z], output=h1,
                                           gradients_to_update=[None, None],
                                           backward_operation=sub_backward)
        h2 = h1 * self.n
        self.autograd_engine.add_operation(inputs=[h1, self.n], output=h2,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        h3 = self.z * self.hidden
        self.autograd_engine.add_operation(inputs=[self.z, self.hidden], output=h3,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        h_t = h2 + h3
        self.autograd_engine.add_operation(inputs=[h2, h3], output=h_t,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        
        assert self.x.shape == (self.input_size,)
        assert self.hidden.shape == (self.hidden_size,)
        assert self.r.shape == (self.hidden_size,)
        assert self.z.shape == (self.hidden_size,)
        assert self.n.shape == (self.hidden_size,)
        assert h_t.shape == (self.hidden_size,) # h_t is the final output of you GRU cell.

        return h_t
