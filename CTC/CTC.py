import numpy as np
import sys
sys.path.append("./")
from mytorch.autograd_engine import *
from mytorch.functional import *

class CTC(object):
    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        skip_connect = [0]
        for i, symbol in enumerate(target):
            extended_symbols.append(symbol)
            skip_connect.append(int(i > 0 and target[i] != target[i-1]))
            extended_symbols.append(self.BLANK)
            skip_connect.append(0)

        N = len(extended_symbols)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextended_symbols[i]]

        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skip_connect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        for t in range(1, T):
            alpha[t][0] = alpha[t - 1][0] * logits[t][extended_symbols[0]]
            for l in range(1, S):
                if bool(skip_connect[l]):
                    alpha[t][l] = alpha[t-1][l] + \
                        alpha[t-1][l-1] + alpha[t-1][l-2]
                else:
                    alpha[t][l] = alpha[t - 1][l] + alpha[t-1][l-1]
                alpha[t][l] *= logits[t][extended_symbols[l]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extended_symbols[i]]

        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------

        beta[-1, -1] = 1
        beta[-1, -2] = 1

        for t in range(T - 2, -1, -1):
            beta[t][-1] = beta[t + 1][-1] * logits[t + 1][extended_symbols[-1]]
            for i in range(S - 2, -1, -1):
                if i + 2 < S - 1 and bool(skip_connect[i + 2]):
                    beta[t][i] += beta[t + 1][i] * \
                        logits[t + 1][extended_symbols[i]]
                    beta[t][i] += beta[t + 1][i + 1] * \
                        logits[t + 1][extended_symbols[i + 1]]
                    beta[t][i] += beta[t + 1][i + 2] * \
                        logits[t + 1][extended_symbols[i + 2]]
                else:
                    beta[t][i] += beta[t + 1][i] * \
                        logits[t + 1][extended_symbols[i]]
                    beta[t][i] += beta[t + 1][i + 1] * \
                        logits[t + 1][extended_symbols[i + 1]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------

        gamma += alpha * beta
        sumgamma += np.sum(gamma, axis=1)
        return gamma / sumgamma.reshape(-1, 1)


class CTCLoss(object):

    def __init__(self, autograd_engine, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()
        self.autograd_engine = autograd_engine

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        
        # NOTE: Toggle using ctc_loss_backward version 
        # or a version using more primitive operations
        self.USE_PRIMITIVE = False
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        if self.USE_PRIMITIVE:
            return self.forward_primitive(logits, target, input_lengths, target_lengths)
        else:
            return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        # IMP:
        # Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.gammas = np.empty(B, dtype=object)
        self.extended_symbols = np.empty(B, dtype=object)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target_t = target[batch_itr, :target_lengths[batch_itr]]
            logits_t = self.logits[:input_lengths[batch_itr], batch_itr]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_t)
            alpha = self.ctc.get_forward_probs(logits_t, extended_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logits_t, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            
            for r in range(gamma.shape[1]):
                total_loss[batch_itr] -= np.sum(gamma[:, r] * np.log(logits_t[:, extended_symbols[r]]))
                
            
            
            self.gammas[batch_itr] = gamma
            self.extended_symbols[batch_itr] = extended_symbols

        total_loss = np.sum(total_loss) / B

        self.autograd_engine.add_operation(inputs=[self.logits, input_lengths, self.gammas, self.extended_symbols],
                                           output= total_loss,
                                           gradients_to_update=[None, None, None, None],
                                           backward_operation=ctc_loss_backward)     
        
        
        return total_loss
    
    def forward_primitive(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        # IMP:
        # Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        # NOTE: Since, arrays with the same views cannot be added to the gradient buffer
        # updating loss must be kept track of in this manner
        tmp_loss = {k:[np.array([0.0], dtype=np.float64)] for k in range(B)}
        self.gammas = np.empty(B, dtype=object)
        self.extended_symbols = np.empty(B, dtype=object)
        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target_t = target[batch_itr, :target_lengths[batch_itr]]
            logits_t = self.logits[:input_lengths[batch_itr], batch_itr]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_t)
            alpha = self.ctc.get_forward_probs(logits_t, extended_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logits_t, extended_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)


            for r in range(gamma.shape[1]):

                tmp_loss[batch_itr].append(np.array([0.0], dtype=np.float64))

                idx1 = np.index_exp[:input_lengths[batch_itr], batch_itr, extended_symbols[r]]
                logits_r = self.logits[idx1].copy(order='A')
                self.autograd_engine.add_operation(inputs=[self.logits, np.array(idx1, dtype=object)],
                                                   output=logits_r,
                                                   gradients_to_update=[None, None],
                                                   backward_operation=slice_backward)
                
                i1 = np.log(logits_r)
                self.autograd_engine.add_operation(inputs=[logits_r],
                                                   output=i1,
                                                   gradients_to_update=[None],
                                                   backward_operation=log_backward)
                
                gamma_r = gamma[:, r].copy()
                i2 = i1 * gamma_r
                self.autograd_engine.add_operation(inputs=[i1, gamma_r],
                                                   output=i2,
                                                   gradients_to_update=[None, None],
                                                   backward_operation=mul_backward)

                i3 = np.sum(i2, keepdims=True)
                self.autograd_engine.add_operation(inputs=[i2],
                                                   output=i3,
                                                   gradients_to_update=[None],
                                                   backward_operation=sum_backward)

                tmp_loss[batch_itr][-1] = tmp_loss[batch_itr][-2] - i3
                self.autograd_engine.add_operation(inputs=[tmp_loss[batch_itr][-2], i3],
                                                   output=tmp_loss[batch_itr][-1],
                                                   gradients_to_update=[None, None],
                                                   backward_operation=sub_backward)
                
                
            assert(len(tmp_loss[batch_itr]) == gamma.shape[1] + 1)


            self.gammas[batch_itr] = gamma
            self.extended_symbols[batch_itr] = extended_symbols

        # NOTE: Again, since arrays with the same views cannot be added to the gradient buffer
        # summing the loss must be done in this manner to that the operations wrt to each
        # element can be tracked for gradient backprop
        total_loss = [np.array([0.0], dtype=np.float64) for _ in range(B+1)]
        for i in range(B):
            total_loss[i+1] = total_loss[i] + tmp_loss[i][-1]
            self.autograd_engine.add_operation(inputs=[total_loss[i], tmp_loss[i][-1]],
                                               output=total_loss[i+1],
                                               gradients_to_update=[None, None],
                                               backward_operation=add_backward)

        # NOTE: Dont add div operation to match test values
        i4 = total_loss[-1] / B
        return i4