import numpy as np
import numpy.typing as npt
from absl import app, logging

from tic_tac_gpt.model.optimal_model import OptimalModel


def entropy(p: npt.NDArray) -> float:
    return -np.sum(p * np.log(p, out=np.zeros_like(p), where=p != 0)).item()


def kl_divergence(p: npt.NDArray, q: npt.NDArray) -> float:
    p[q == 0] = 0
    return np.sum(p * np.log(p / q, out=np.zeros_like(p), where=p != 0)).item()


def main(_):
    opt_model = OptimalModel()

    total_weight = 0
    total_loss = 0
    for prefix in opt_model:
        probs, weight = opt_model[prefix]
        total_loss += entropy(probs) * weight
        total_weight += weight
    logging.info("Optimal model entropy: %.6f nats", total_loss / total_weight)

    # calculate chance-level kl divergence
    p = np.zeros(())
    for prefix in opt_model:
        probs, weight = opt_model[prefix]
        p = p + probs * weight
    p /= total_weight
    logging.info("Average p: %s", p)

    kl = 0
    for prefix in opt_model:
        probs, weight = opt_model[prefix]
        kl += kl_divergence(probs, p)
    logging.info("Chance-level KL divergence: %.6f nats", kl / len(opt_model))

    acc = 0
    for prefix in opt_model:
        probs, weight = opt_model[prefix]
        if probs[np.argmax(p)] > 0:
            acc += 1
    logging.info("Accuracy: %.2f%%", acc * 100 / len(opt_model))


if __name__ == "__main__":
    app.run(main)
