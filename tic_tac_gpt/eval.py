import json
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
from absl import app, flags, logging
from lightning import fabric
from ml_tools.itertools_ import batch
from ml_tools.metrics import metrics
from scipy.special import rel_entr
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from tic_tac_gpt.data import TicTacToeDataset
from tic_tac_gpt.model.optimal_model import OptimalModel

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "out/model/exp1", "Checkpoint file")
flags.DEFINE_integer("step", None, "Step to evaluate")
flags.DEFINE_multi_float("prune_percent", [], "Percentage of neurons to prune")


def normalize(x, dim=-1):
    x = x - x.mean(dim, keepdim=True)
    scale = (x.pow(2).mean(dim, keepdim=True)).sqrt()
    return x / scale


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return rel_entr(p, q).sum(axis=-1)


@torch.no_grad()
def evaluate_model(forwards: list, opt_model: OptimalModel, device):
    for prefixes in batch(tqdm(opt_model), 4096):
        x = torch.nested.nested_tensor(
            prefixes, dtype=torch.long, device=device
        ).to_padded_tensor(TicTacToeDataset.pad_token)
        all_preds = [forward(x).softmax(-1).cpu().numpy() for forward in forwards]

        indices = np.array([len(p) - 1 for p in prefixes])
        arange = np.arange(len(prefixes))
        all_preds = [preds[arange, indices] for preds in all_preds]

        for i, prefix in enumerate(prefixes):
            probs, weight = opt_model[prefix]

            for j, preds in enumerate(all_preds):
                arg_max = np.argmax(preds[i])
                kl = kl_divergence(probs, preds[i]) * weight / opt_model.total_weight
                assert kl >= 0, (probs, preds[i])
                is_valid = probs[arg_max] > 0
                is_acc = (probs <= probs[arg_max]).all()

                metrics.log_dict(
                    {
                        f"{j}/kl": kl.item(),
                        f"{j}/valid": is_valid.item(),
                        f"{j}/accurate": is_acc.item(),
                    }
                )

    kls = []
    valids = []
    accs = []
    for j in range(len(forwards)):
        logs = metrics.collect_group(f"{j}/")
        kl = np.array(logs[f"{j}/kl"]).sum().item()
        valid = np.array(logs[f"{j}/valid"]).mean().item()
        acc = np.array(logs[f"{j}/accurate"]).mean().item()
        kls.append(kl)
        valids.append(valid)
        accs.append(acc)

    return kls, valids, accs


def main(_):
    checkpoint_dir = Path(FLAGS.checkpoint)
    with open(checkpoint_dir / "config.pkl", "rb") as f:
        config: HookedTransformerConfig = pickle.load(f)
    model = HookedTransformer(config)

    F = fabric.Fabric(precision="16-mixed")
    state_dict = F.load(
        checkpoint_dir
        / ("model.pt" if FLAGS.step is None else f"model_{FLAGS.step}.pt")
    )
    model.load_and_process_state_dict(state_dict)
    model.eval()

    opt_model = OptimalModel()

    methods = []

    # --- Full model ---
    def forward_full(x):
        return model(x)

    # methods.append((forward_full, "full"))

    # --- MLP only ---
    def forward_mlp_only(x):
        logits, cache = model.run_with_cache(x)
        return model.unembed(normalize(cache["mlp_out", -1]))

    # methods.append((forward_mlp_only, "mlp_only"))

    # --- MLP only, no layer normalization ---
    def forward_mlp_only_no_ln(x):
        logits, cache = model.run_with_cache(x)
        return model.unembed(cache["mlp_out", -1])

    # methods.append((forward_mlp_only_no_ln, "mlp_only_no_ln"))

    # --- Pruned model ---
    neurons_file = checkpoint_dir / f"neurons_{FLAGS.step}.json"
    if neurons_file.is_file():
        with open(neurons_file, "r") as f:
            neurons = {int(k): v for k, v in json.load(f).items()}
        neurons = sorted(neurons.keys(), key=lambda k: -neurons[k])

        def neuron_hook(value, hook, ignored_neurons):
            value[:, :, ignored_neurons] -= 100
            return value

        for prune_percent in FLAGS.prune_percent:
            ignored_neurons = neurons[: int(len(neurons) * prune_percent)]
            logging.info(f"Ignoring {len(ignored_neurons)} neurons")

            methods.append(
                (
                    partial(
                        model.run_with_hooks,
                        fwd_hooks=[
                            (
                                utils.get_act_name("mlp_pre", 0),
                                partial(neuron_hook, ignored_neurons=ignored_neurons),
                            )
                        ],
                    ),
                    f"pruned_{prune_percent}",
                )
            )

    # --- Chance level ---
    # chance_level_logits = torch.tensor(0, dtype=torch.float32)
    # for prefix in opt_model:
    #     probs, weight = opt_model[prefix]
    #     chance_level_logits += probs * weight / opt_model.total_weight
    # chance_level_logits = torch.log(chance_level_logits)

    # def forward_chance(x):
    #     return torch.broadcast_to(
    #         chance_level_logits, (x.shape[0], x.shape[1], chance_level_logits.shape[0])
    #     )

    # methods.append((forward_chance, "chance"))

    forwards, names = zip(*methods)
    kls, valids, accs = evaluate_model(forwards, opt_model, F.device)
    for name, kl, valid, acc in zip(names, kls, valids, accs):
        logging.info(
            "%s: KL divergence: %.6f nats, valid accuracy: %.2f%%, accurate accuracy: %.2f%%",
            name,
            kl,
            valid * 100,
            acc * 100,
        )


if __name__ == "__main__":
    app.run(main)
