import json
import pickle
from pathlib import Path

import torch
from absl import app, flags, logging
from lightning import fabric
from ml_tools.metrics import metrics
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tic_tac_gpt.data import TicTacToeDataset
from tic_tac_gpt.data.build_tic_tac_toe import TicTacToeState

FLAGS = flags.FLAGS
flags.DEFINE_string("train_file", "out/dataset/50_50/train.jsonl", "Train file")
flags.DEFINE_string("test_file", "out/dataset/50_50/test.jsonl", "Test file")
flags.DEFINE_string("checkpoint", "out/model/exp1", "Checkpoint file")


def main(_):
    checkpoint_dir = Path(FLAGS.checkpoint)
    ds_train = TicTacToeDataset(FLAGS.train_file)
    ds_test = TicTacToeDataset(FLAGS.test_file)
    with open(checkpoint_dir / "config.pkl", "rb") as f:
        config: HookedTransformerConfig = pickle.load(f)
    model = HookedTransformer(config)

    F = fabric.Fabric(precision="16-mixed")
    state_dict = F.load(checkpoint_dir / "model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    @torch.no_grad()
    def eval_batch(batch):
        (x,) = batch
        b, s = x.shape
        probs = model(x[:, :-1], return_type="logits").softmax(dim=-1)
        preds = model(x[:, :-1], return_type="logits").argmax(dim=-1)

        x = x.cpu().tolist()
        preds = preds.cpu().tolist()
        valid_pred = []
        for x_batch, preds_batch, probs_batch in zip(x, preds, probs):
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

                # if not valid_pred_batch[i - 1]:
                #     print(
                #         f"{prefix=} {next_move=} {preds_batch[i-1]=} {probs_batch[i-1]=} {game_state.result=} {valid_moves=} {valid_pred_batch[i - 1]=}"
                #     )
                #     print(game_state)
                #     breakpoint()
        return torch.tensor(valid_pred)

    def eval_ds(ds):
        for batch in DataLoader(ds, batch_size=512):
            valid_pred = eval_batch(batch)
            metrics.log_dict({"valid": valid_pred})
        logs = metrics.collect("valid")
        acc = torch.cat(logs["valid"]).float().mean().item()
        return acc

    train_acc = eval_ds(ds_train)
    logging.info("Train accuracy: %.2f%%", train_acc * 100)
    test_acc = eval_ds(ds_test)
    logging.info("Test accuracy: %.2f%%", test_acc * 100)


if __name__ == "__main__":
    app.run(main)
