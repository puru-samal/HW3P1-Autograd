# HW3P1-Autograd

An implementation of HW3P1 using the autograd-integrated myTorch Library. Since, Autograd1 involved creating Linear layers with autograd support to do backprop, and since Prof. stated that the motivation behind providing students the option to do HW3P1 was to simplify/abstract some of the calculations required for the BPTT and CTCLoss, I've taken a more modular approach to this assignment where:

- `RNNCell` is composed of `nn.Linear`'s
- `GRUCell` is composed of RNNCell's with an optional scale factor (`scale_h`) for the hidden linear transformation.
- `CTCLoss` forward is done with tracking primitive operations so that gradients are automatically backpropagated without an needing explicit support for ctc_loss_backward. However, this can get somewhat involved due to there being no support in the Autograd library to accomodate numpy.ndarrays sharing the same base memory location. So, essentially one must make sure that appropriate copies while doing numpu in-place operations so that the inputs and outputs are tracked in the computation graph. An 'non-primitive' approach by writing a `ctc_loss_backward` function might be an easier alternative. Regardless, The `CTCLoss` class in `./CTC/CTC.py` has implementation for both approaches in the `forward` and `forward_primitive` methods whose use can be toggled by setting the `USE_PRIMITIVE` class variable.
- Testcases required changes to accomodate the implementation/interface differences in the Autograd version of myTorch. The general approach taken to test correctness of tests requiring `RNNCell` and `GRUCell` was to run forward and backward network passes and compare outputs and gradients of network parameters with an equivalent `PyTorch` network implementation. Other testcase changes are more minor, mostly only requiring instantiation of an `autograd_engine.Autograd` object, passing the instance to the class requiring operation-tracking and calling it's `backward` method when gradients need to be backpropagated. See additional `NOTE:` comments in the `test_xx.py` files for wny a particular test case required changes.

**Taking the original HW3P1 handout and modifying it all the way to accomodate the Autograd version was a quite tedious. For this reason, I recommend providing some starter code to students wishing to attempt the assignment with Autograd. This implementation might provide a useful starting point to what that can look like. There are comments prepended with `NOTE:` highlighting key implementation details between the original HW3P1 and HW3P1-Autograd which might be worth repurposing to comments in the starter code.**

# Autograd Extensions

The following extensions to the Autograd1 library were required on my end to complete the assignment. They were progressively-implemented while going through the assignment, but it might be worth considering providing some or parts of these extensions upfront.

## in `./mytorch/functional.py`:

- Support for handling broadcasting during backprop required (see: `unbroadcast`)
- `sum_backward`: backward operation for np.sum(x, axis=0). Used in:
  - `./autograder/test_rnn.py`
  - `./autograder/test_rnn_toy.py`
  - `./CTC/CTC.py`
- `tanh_backward`: modified to keep track of hidden state for BPTT as done in HW3P1. The `Tanh` function in `./mytorch/nn/activation.py`
  will also require a change to accomodate this.
- `expand_dims_backward`: backward operation for np.expand_dims(x, axis=i). Used in Linear which forms the building blocks of RNNCell and GRUCell class. Required to match PyTorch shape constraints and pass some shape correctness tests. Used in:
  - `./mytorch/nn/linear.py`
  - `./autograder/test_gru_toy`
  - `./autograder/test_gru.py`
- `squeeze_backward`: backward operation for np.squeeze(x, axis=i). Required to match PyTorch shape constraints and pass some shape correctness tests. Used in:
  - `./mytorch/nn/linear.py`
- `slice_backward`: backward operation for x[indices]. Used in:
  - `./autograder/test_rnn.py`
  - `./autograder/test_rnn_toy.py`
  - `./CTC/CTC.py`
- `ctc_loss_backward`: (Optional) if going the 'non-primitive' route.

## in `./mytorch/linear.py`:

After implementing broadcasting, the shape of bias in `Linear` can be modified to `(out_features,)` instead of `(out_features, 1)` to meet the shape requirements and assertions in the original HW3P1 starter code and shape-match `PyTorch` outputs. Also, handling batched [`(N, feature_dim)`] verses un-batched inputs [`(feature_dim,)`] here might save you a lot of trouble down the road.

# Tests

- Run `toy_runner.py` with `python autograder/toy_runner.py 'test_name'` or `python autograder/toy_runner.py`
- Run `runner.py` with `python autograder/runner.py 'test_name'` or `python autograder/runner.py`
