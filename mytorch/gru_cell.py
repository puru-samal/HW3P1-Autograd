import numpy as np
from .nn.activation import *
from .nn.linear import *
from mytorch.functional import *
from mytorch.autograd_engine import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size, autograd_engine):
        self.d = input_size
        self.h = hidden_size
        self.autograd_engine = autograd_engine
        h = self.h
        d = self.d
        self.x_t = 0
        
        self.rx = Linear(input_size, hidden_size, self.autograd_engine)
        self.zx = Linear(input_size, hidden_size, self.autograd_engine)
        self.nx = Linear(input_size, hidden_size, self.autograd_engine)

        self.rh = Linear(hidden_size, hidden_size, self.autograd_engine)
        self.zh = Linear(hidden_size, hidden_size, self.autograd_engine)
        self.nh = Linear(hidden_size, hidden_size, self.autograd_engine)

        # Init Weights
        self.rx.W = np.random.randn(h, d)
        self.zx.W = np.random.randn(h, d)
        self.nx.W = np.random.randn(h, d)

        self.rh.W = np.random.randn(h, h)
        self.zh.W = np.random.randn(h, h)
        self.nh.W = np.random.randn(h, h)

        self.rx.b = np.random.randn(h)
        self.zx.b = np.random.randn(h)
        self.nx.b = np.random.randn(h)

        self.rh.b = np.random.randn(h)
        self.zh.b = np.random.randn(h)
        self.nh.b = np.random.randn(h)

        # Init Gradients
        self.rx.dW = np.zeros((h, d))
        self.zx.dW = np.zeros((h, d))
        self.nx.dW = np.zeros((h, d))

        self.rh.dW = np.zeros((h, h))
        self.zh.dW = np.zeros((h, h))
        self.nh.dW = np.zeros((h, h))

        self.rx.db = np.zeros((h))
        self.zx.db = np.zeros((h))
        self.nx.db = np.zeros((h))

        self.rh.db = np.zeros((h))
        self.zh.db = np.zeros((h))
        self.nh.db = np.zeros((h))

        self.r_act = Sigmoid(self.autograd_engine)
        self.z_act = Sigmoid(self.autograd_engine)
        self.h_act = Tanh(self.autograd_engine)

        # Define other variables to store forward results for backward here
        self.r = None
        self.z = None
        self.n = None


    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.rx.W = Wrx
        self.zx.W = Wzx
        self.nx.W = Wnx
        self.rh.W = Wrh
        self.zh.W = Wzh
        self.nh.W = Wnh
        self.rx.b = brx
        self.zx.b = bzx
        self.nx.b = bnx
        self.rh.b = brh
        self.zh.b = bzh
        self.nh.b = bnh

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

        # NOTE: r = σ (W_ir x + b_ir + W_hr h + b_hr)
        r1 = self.rx(self.x)
        r2 = self.rh(self.hidden)
        r3 = r1 + r2
        self.autograd_engine.add_operation(inputs=[r1, r2], output=r3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        self.r = self.r_act(r3)

        # NOTE: z = σ (W_iz x + b_iz + W_hz h + b_hz)
        z1 = self.zx(x)
        z2 = self.zh(self.hidden)
        z3 = z1 + z2
        self.autograd_engine.add_operation(inputs=[z1, z2], output=z3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        self.z = self.z_act(z3)

        # NOTE: n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        n1 = self.nx(x)
        n2_ = self.nh(self.hidden)
        n2 = self.r * n2_
        self.autograd_engine.add_operation(inputs=[self.r, n2_], output=n2,
                                           gradients_to_update=[None, None],
                                           backward_operation=mul_backward)
        n3 = n1 + n2
        self.autograd_engine.add_operation(inputs=[n1, n2], output=n3,
                                           gradients_to_update=[None, None],
                                           backward_operation=add_backward)
        self.n = self.h_act(n3)

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
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
