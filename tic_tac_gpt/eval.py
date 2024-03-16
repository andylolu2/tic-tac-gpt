import pickle
from pathlib import Path

import torch
from absl import app, flags, logging
from lightning import fabric
from ml_tools.itertools_ import batch
from ml_tools.metrics import metrics
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tic_tac_gpt.data import TicTacToeDataset
from tic_tac_gpt.model.optimal_model import OptimalModel

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "out/model/exp1", "Checkpoint file")
flags.DEFINE_integer("step", None, "Step to evaluate")


def normalize(x, dim=-1):
    x = x - x.mean(dim, keepdim=True)
    scale = (x.pow(2).mean(dim, keepdim=True)).sqrt()
    return x / scale


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

    @torch.no_grad()
    def forward(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, cache = model.run_with_cache(x)
        mlp_out = cache["mlp_out", -1]
        logits_mlp_only = model.unembed(normalize(mlp_out))
        logits_mlp_only_no_ln = model.unembed(mlp_out)
        return logits, logits_mlp_only, logits_mlp_only_no_ln  # type: ignore

    opt_model = OptimalModel()

    chance_level_logits = torch.tensor(0, dtype=torch.float32)
    total_weight = 0
    for prefix in opt_model:
        probs, weight = opt_model[prefix]
        chance_level_logits += probs * weight
        total_weight += weight
    chance_level_logits /= total_weight
    chance_level_logits = torch.log(chance_level_logits)

    def eval_batch(prefixes, logits):
        kls = []
        valid = []
        accurate = []
        for prefix, logit in zip(prefixes, logits):
            probs, weight = opt_model[prefix]
            probs = torch.tensor(probs, dtype=torch.float32)
            pred = logit.argmax()

            kl = kl_divergence(probs, logit) * (weight / total_weight)
            is_valid = probs[pred] > 0
            is_acc = (probs <= probs[pred]).all()

            kls.append(kl)
            valid.append(is_valid)
            accurate.append(is_acc)

        return torch.tensor(kls), torch.tensor(valid), torch.tensor(accurate)

    for prefixes in batch(opt_model, 4096):
        x = torch.nested.nested_tensor(
            prefixes, dtype=torch.long, device=F.device
        ).to_padded_tensor(TicTacToeDataset.pad_token)
        logits, logits_mlp_only, logits_mlp_only_no_ln = forward(x)
        logits = logits.to("cpu")
        logits_mlp_only = logits_mlp_only.to("cpu")
        logits_mlp_only_no_ln = logits_mlp_only_no_ln.to("cpu")

        indices = torch.tensor([len(p) - 1 for p in prefixes], dtype=torch.long)
        arange = torch.arange(len(prefixes), dtype=torch.long)
        logits = logits[arange, indices]
        logits_mlp_only = logits_mlp_only[arange, indices]
        logits_mlp_only_no_ln = logits_mlp_only_no_ln[arange, indices]

        kls, valid, accurate = eval_batch(prefixes, logits)
        kls_mlp_only, valid_mlp_only, accurate_mlp_only = eval_batch(
            prefixes, logits_mlp_only
        )
        kls_mlp_only_no_ln, valid_mlp_only_no_ln, accurate_mlp_only_no_ln = eval_batch(
            prefixes, logits_mlp_only_no_ln
        )
        kls_chance, valid_chance, accurate_chance = eval_batch(
            prefixes, torch.tile(chance_level_logits, (len(prefixes), 1))
        )
        metrics.log_dict(
            {
                "kl_div": kls,
                "kl_div_mlp_only": kls_mlp_only,
                "kl_div_mlp_only_no_ln": kls_mlp_only_no_ln,
                "kl_div_chance": kls_chance,
                "valid": valid,
                "valid_mlp_only": valid_mlp_only,
                "valid_mlp_only_no_ln": valid_mlp_only_no_ln,
                "valid_chance": valid_chance,
                "accurate": accurate,
                "accurate_mlp_only": accurate_mlp_only,
                "accurate_mlp_only_no_ln": accurate_mlp_only_no_ln,
                "accurate_chance": accurate_chance,
            }
        )
    logs = metrics.collect_group("")
    logging.info(
        "KL divergence: %.6f nats (MLP only: %.6f nats, No LN: %.6f nats, chance: %.6f nats)",
        torch.cat(logs["kl_div"]).sum().item(),
        torch.cat(logs["kl_div_mlp_only"]).sum().item(),
        torch.cat(logs["kl_div_mlp_only_no_ln"]).sum().item(),
        torch.cat(logs["kl_div_chance"]).sum().item(),
    )

    valid = torch.cat(logs["valid"]).float().mean().item()
    valid_mlp_only = torch.cat(logs["valid_mlp_only"]).float().mean().item()
    valid_mlp_only_no_ln = torch.cat(logs["valid_mlp_only_no_ln"]).float().mean().item()
    valid_chance = torch.cat(logs["valid_chance"]).float().mean().item()
    logging.info(
        "Valid accuracy: %.2f%% (MLP only: %.2f%%, No LN: %.2f%%, chance: %.2f%%)",
        valid * 100,
        valid_mlp_only * 100,
        valid_mlp_only_no_ln * 100,
        valid_chance * 100,
    )

    acc = torch.cat(logs["accurate"]).float().mean().item()
    acc_mlp_only = torch.cat(logs["accurate_mlp_only"]).float().mean().item()
    acc_mlp_only_no_ln = (
        torch.cat(logs["accurate_mlp_only_no_ln"]).float().mean().item()
    )
    acc_chance = torch.cat(logs["accurate_chance"]).float().mean().item()
    logging.info(
        "Accurate accuracy: %.2f%% (MLP only: %.2f%%, No LN: %.2f%%, chance: %.2f%%)",
        acc * 100,
        acc_mlp_only * 100,
        acc_mlp_only_no_ln * 100,
        acc_chance * 100,
    )


if __name__ == "__main__":
    app.run(main)
