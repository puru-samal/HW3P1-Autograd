import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[1])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        for t in range(len(y_probs[1])):
            path_prob *= np.max(y_probs[:, t, 0])
            idx = np.argmax(y_probs[:, t, 0])
            if idx != blank:
                if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[idx-1]:
                    decoded_path.append(self.symbol_set[idx-1])

        return "".join(decoded_path), path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        BestPaths = {'-': 1.0}
        tempBestPathWithScores = {}

        for t in range(T):
            sym_probs = y_probs[:, t, 0]

            if len(BestPaths) > self.beam_width:
                BestPaths = dict(sorted(
                    BestPaths.items(), key=lambda item: item[1], reverse=True)[:self.beam_width])

            for path, score in BestPaths.items():
                for sym in range(y_probs.shape[0]):

                    new_sym = '-' if sym == 0 else self.symbol_set[sym-1]

                    new_path = ''

                    if new_sym == path[-1]:
                        if path[-1] == '-':   # Case1: new_sym == prev and prev == blank
                            new_path = path   # Dont extend

                        else:                 # Case2: new_sym == prev and prev != blank
                            new_path = path   # Don't extend

                    else:
                        if path[-1] == '-':   # Case3: new_sym != prev and prev == blank
                            # Drop blank and extend
                            new_path = path[:-1] + new_sym

                        else:  # Case4: new_sym != prev and prev != blank
                            new_path = path + new_sym

                    if new_path in tempBestPathWithScores:
                        tempBestPathWithScores[new_path] += sym_probs[sym] * score
                    else:
                        tempBestPathWithScores[new_path] = sym_probs[sym] * score

            BestPaths = tempBestPathWithScores  # Update
            tempBestPathWithScores = {}        # Reset

        mergedPathScores = {}
        bestPath = '-'  # Blank symbol path
        FinalPathScore = 0

        for path, score in BestPaths.items():

            if path[-1] == '-':  # Remove ending blank
                path = path[:-1]

            # Update scores for translated path
            if path in mergedPathScores:
                mergedPathScores[path] += score
            else:
                mergedPathScores[path] = score

            score = mergedPathScores[path]

            # Update bestPath and bestScores
            if score > FinalPathScore:
                bestPath = path
                FinalPathScore = score

        return bestPath, mergedPathScores
