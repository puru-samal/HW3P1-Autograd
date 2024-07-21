# HW3P1-Autograd

An implementation of HW3P1 using the autograd-integrated myTorch Library. Below I discuss the steps I had to go through, sorted in a manner that I find most logical, in order to complete the assignment.

# Autograd Extensions

The following extensions to the autograd library were required on my end to complete the assignment:

## in `mytorch/functional.py`:

- Support for handling broadcasting during backprop required (see: `unbroadcast`)
- `sum_backward`: backward operation for np.sum(x, axis=0). Used in:
  - `./autograder/test_rnn.py:140`
  - `./autograder/test_rnn_toy.py:136`
  - `./CTC/CTC.py:402`
- `tanh_backward`: modified to keep track of hidden state for BPTT as done in HW3P1
- `expand_dims_backward`: backward operation for np.expand_dims(x, axis=i). Used in Linear which forms the building blocks of RNNCell and GRUCell class. Required to match PyTorch shape constraints and pass some shape correctness tests. Used in:
  - `./mytorch/nn/linear.py:50`
- `squeeze_backward`: backward operation for np.squeeze(x, axis=i). Required to match PyTorch shape constraints and pass some shape correctness tests. Used in:
  - `./mytorch/nn/linear.py:73`
- `slice_backward`: backward operation for x[indices]. Used in:
  - `./autograder/test_rnn.py:124`
  - `./autograder/test_rnn.py:144`
  - `./autograder/test_rnn_toy.py:118`
  - `./autograder/test_rnn_toy.py:140`
  - `./CTC/CTC.py:386`
- `ctc_loss_backward`: (Optional) Tracking each primitive function to do the backward pass in CTCLoss can be very involved. Also, I think PyTorch writes it's own ctc_loss_backward function instead of manually tracking each primitive function to do the backward pass. Regardless, The `CTCLoss` class in `./CTC/CTC.py` has implementation for both approaches in the `forward` and `forward_primitive` methods whose use can be toggled by setting the `USE_PRIMITIVE` class variable.

## in `mytorch/linear.py`:
