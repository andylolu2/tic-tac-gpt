from collections import defaultdict

import numpy as np
from absl import logging

from tic_tac_gpt.data import TicTacToeDataset, TicTacToeState


class OptimalModel:
    def __init__(self):
        self.ds = TicTacToeDataset.from_states(TicTacToeState.all_games())

        self.table = defaultdict(lambda: np.zeros(self.ds.vocab_size))
        self.weights = {}

        for (x,) in self.ds:
            x = tuple(x.tolist())
            for i in range(1, len(x)):
                if x[i] == self.ds.pad_token:
                    break
                prefix = x[:i]
                self.table[prefix][x[i]] += 1

        logging.info("Counted %d prefixes", len(self.table))
        logging.info("Game counts from [BOS]: %s", self.table[(self.ds.bos_token,)])

        self.total_weight = sum(np.sum(counts) for counts in self.table.values())
        for prefix, counts in self.table.items():
            self.weights[prefix] = np.sum(counts)
            self.table[prefix] = counts / self.weights[prefix]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, prefix):
        return self.table[prefix], self.weights[prefix]

    def __iter__(self):
        return iter(self.table)
