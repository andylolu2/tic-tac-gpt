import json
import pickle
from pathlib import Path

import torch
from absl import app, flags, logging
from lightning import fabric
from ml_tools.itertools_ import batch
from ml_tools.metrics import metrics
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tic_tac_gpt.data import TicTacToeDataset, TicTacToeState
from tic_tac_gpt.model.optimal_model import OptimalModel

FLAGS = flags.FLAGS
flags.DEFINE_string("train_file", "out/dataset/50_50/train.jsonl", "Train file")
flags.DEFINE_string("test_file", "out/dataset/50_50/test.jsonl", "Test file")
flags.DEFINE_string("checkpoint", "out/model/exp1", "Checkpoint file")
flags.DEFINE_integer("step", None, "Step to evaluate")


def main(_):
    checkpoint_dir = Path(FLAGS.checkpoint)
    ds_train = TicTacToeDataset.from_file(FLAGS.train_file)
    ds_test = TicTacToeDataset.from_file(FLAGS.test_file)
    with open(checkpoint_dir / "config.pkl", "rb") as f:
        config: HookedTransformerConfig = pickle.load(f)
    model = HookedTransformer(config)

    F = fabric.Fabric(precision="16-mixed")
    state_dict = F.load(
        checkpoint_dir
        / ("model.pt" if FLAGS.step is None else f"model_{FLAGS.step}.pt")
    )
    model.load_state_dict(state_dict)
    model.eval()

    @torch.no_grad()
    def forward(x) -> tuple[torch.Tensor, torch.Tensor]:
        logits, cache = model.run_with_cache(x)
        mlp_out = cache["mlp_out", -1]
        mlp_out = cache.apply_ln_to_stack(mlp_out)
        logits_mlp_only = model.unembed(mlp_out)
        return logits, logits_mlp_only  # type: ignore

    def eval_batch(batch):
        (x,) = batch
        b, s = x.shape

        logits, logits_mlp_only = forward(x[:, :-1])
        preds = logits.argmax(dim=-1)
        preds_mlp_only = logits_mlp_only.argmax(dim=-1)

        def count_valid(x, preds):
            x = x.cpu().tolist()
            preds = preds.cpu().tolist()
            valid_pred = []
            for x_batch, preds_batch in zip(x, preds):
                for i in range(1, s):
                    prefix, next_move = x_batch[1:i], x_batch[i]
                    if next_move == TicTacToeDataset.pad_token:
                        continue

                    game_state = TicTacToeState(prefix)
                    if game_state.result != "in_progress":
                        valid_moves = {TicTacToeDataset.encode_one(game_state.result)}
                    else:
                        valid_moves = set(game_state.next_moves())
                    valid_pred.append(preds_batch[i - 1] in valid_moves)
            return torch.tensor(valid_pred)

        metrics.log_dict(
            {
                "valid": count_valid(x, preds),
                "valid_mlp_only": count_valid(x, preds_mlp_only),
            }
        )

    def eval_ds(ds):
        for batch in DataLoader(ds, batch_size=512):
            eval_batch(batch)
        logs = metrics.collect("valid", "valid_mlp_only")
        acc = torch.cat(logs["valid"]).float().mean().item()
        acc_mlp_only = torch.cat(logs["valid_mlp_only"]).float().mean().item()
        return acc, acc_mlp_only

    train_acc, train_acc_mlp_only = eval_ds(ds_train)
    logging.info(
        "Train accuracy: %.2f%% (MLP only: %.2f%%)",
        train_acc * 100,
        train_acc_mlp_only * 100,
    )
    test_acc, test_acc_mlp_only = eval_ds(ds_test)
    logging.info(
        "Test accuracy: %.2f%% (MLP only: %.2f%%)",
        test_acc * 100,
        test_acc_mlp_only * 100,
    )

    opt_model = OptimalModel()

    def kl_div(prefixes, logits):
        kls = []
        for prefix, logit in zip(prefixes, logits):
            probs, _ = opt_model[prefix]
            kls.append(
                torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logit, dim=-1),
                    torch.tensor(probs, dtype=torch.float32, device=logits.device),
                    reduction="sum",
                )
            )
        return torch.tensor(kls)

    for prefixes in batch(opt_model, 4096):
        x = torch.nested.nested_tensor(
            prefixes, dtype=torch.long, device=F.device
        ).to_padded_tensor(TicTacToeDataset.pad_token)
        logits, logits_mlp_only = forward(x)
        indices = torch.tensor(
            [len(p) - 1 for p in prefixes], dtype=torch.long, device=F.device
        )
        arange = torch.arange(len(prefixes), dtype=torch.long, device=F.device)
        logits = logits[arange, indices]
        logits_mlp_only = logits_mlp_only[arange, indices]

        metrics.log_dict(
            {
                "kl_div": kl_div(prefixes, logits),
                "kl_div_mlp_only": kl_div(prefixes, logits_mlp_only),
            }
        )
    logs = metrics.collect("kl_div", "kl_div_mlp_only")
    logging.info(
        "KL divergence: %.6f nats (MLP only: %.6f nats)",
        torch.cat(logs["kl_div"]).sum().item() / len(opt_model),
        torch.cat(logs["kl_div_mlp_only"]).sum().item() / len(opt_model),
    )


if __name__ == "__main__":
    app.run(main)
