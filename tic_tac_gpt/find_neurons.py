import json
import pickle
from pathlib import Path

import torch
from absl import app, flags
from lightning import fabric
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from tic_tac_gpt.data import TicTacToeDataset
from tic_tac_gpt.model.optimal_model import OptimalModel

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "out/model/exp1", "Checkpoint file")
flags.DEFINE_integer("step", None, "Step to evaluate")


def kl_divergence(p: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    return torch.where(
        (p != 0) & (q_logits == -float("inf")),
        float("nan"),
        torch.where(
            p == 0,
            0,
            p * (p.log() - q_logits.log_softmax(dim=-1)),
        ),
    ).sum(dim=-1)


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

    ds = TicTacToeDataset.from_file(Path("out/dataset/50_50_even/games.jsonl"))
    games = ds[torch.randperm(len(ds))[:8192]][0]
    logits = model(games)

    opt_model = OptimalModel()

    def measure_neuron_effect(neuron: int):
        def neuron_hook(value, hook):
            value[:, :, neuron] -= 100
            return value

        patched_logits = model.run_with_hooks(
            games, fwd_hooks=[(utils.get_act_name("mlp_pre", 0), neuron_hook)]
        )

        loss_diffs = []

        for game, logit, patched_logit in zip(
            games.cpu(), logits.cpu(), patched_logits.cpu()
        ):
            mask = (
                (game != TicTacToeDataset.pad_token)
                & (game != TicTacToeDataset.encode_one("[X]"))
                & (game != TicTacToeDataset.encode_one("[O]"))
                & (game != TicTacToeDataset.encode_one("[D]"))
            )
            logit = logit[mask]
            patched_logit = patched_logit[mask]
            game = game[mask].tolist()
            probs = [opt_model[tuple(game[:i])][0] for i in range(1, len(game) + 1)]
            probs = torch.tensor(probs, dtype=torch.float32, device=logit.device)

            loss_diff = (
                kl_divergence(probs, logit) - kl_divergence(probs, patched_logit)
            ).mean()
            loss_diffs.append(loss_diff)

        return torch.stack(loss_diffs).mean().item()

    data = {}
    for neuron in tqdm(range(model.cfg.d_mlp)):
        effect = measure_neuron_effect(neuron)
        data[neuron] = effect

    print(sorted(data.keys(), key=lambda k: data[k]))

    with open(checkpoint_dir / f"neurons_{FLAGS.step}.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    app.run(main)
