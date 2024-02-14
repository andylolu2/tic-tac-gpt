from collections import defaultdict

import numpy as np
from absl import app, flags, logging

from tic_tac_gpt.data import TicTacToeDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_file", "out/dataset/games.jsonl", "Train file")


def entropy(a: np.ndarray):
    p = a / np.sum(a)
    return -np.sum(p * np.log(p, out=np.zeros_like(p), where=p != 0))


def main(_):
    ds = TicTacToeDataset(FLAGS.dataset_file)

    table = defaultdict(lambda: np.zeros(ds.vocab_size))

    for (x,) in ds:
        x = x.tolist()
        for i in range(1, len(x)):
            if x[i] == ds.pad_token:
                break
            prefix = tuple(x[:i])
            table[prefix][x[i]] += 1

    logging.info("Counted %d prefixes", len(table))
    logging.info("Game counts from [BOS]: %s", table[(ds.bos_token,)])

    total_weight = 0
    total_loss = 0
    for prefix, counts in table.items():
        weight = np.sum(counts)
        total_loss += entropy(counts) * weight
        total_weight += weight

    logging.info("Optimal model entropy: %.6f nats", total_loss / total_weight)


if __name__ == "__main__":
    app.run(main)
