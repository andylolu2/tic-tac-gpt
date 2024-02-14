import json

import torch
from absl import app, flags, logging
from lightning import fabric
from ml_tools.infinite_dataloader import InfiniteDataLoader
from ml_tools.metrics import metrics
from ml_tools.torch_module import num_params
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tic_tac_gpt.data import TicTacToeDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("train_file", "out/dataset/train.jsonl", "Train file")
flags.DEFINE_string("test_file", "out/dataset/test.jsonl", "Test file")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("steps", 1000, "Steps")
flags.DEFINE_integer("log_every", 100, "Log every")


def main(_):
    ds_train = TicTacToeDataset(FLAGS.train_file)
    ds_test = TicTacToeDataset(FLAGS.test_file)
    train_loader = InfiniteDataLoader(
        ds_train, batch_size=FLAGS.batch_size, shuffle=True
    )
    test_loader = DataLoader(ds_test, batch_size=FLAGS.batch_size)

    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=2,
            d_model=128,
            d_head=32,
            n_ctx=ds_train.max_seq_len,
            d_vocab=ds_train.vocab_size,
            act_fn="solu_ln",
            attn_only=True,
            normalization_type="LN",
        ),
    )
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    logging.info(f"Config:\n{json.dumps(model.cfg.to_dict(), indent=2, default=str)}")
    logging.info(f"Model has {num_params(model):,} parameters")

    F = fabric.Fabric()
    model, optimizer = F.setup(model, optimizer)
    train_loader, test_loader = F.setup_dataloaders(train_loader, test_loader)
    train_iter = iter(train_loader)

    def train_step(batch):
        (x,) = batch
        logits = model(x, return_type="logits")  # (b s d)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            x[:, 1:].flatten(0, 1),
            ignore_index=ds_train.pad_token,
        )
        F.backward(loss)
        optimizer.step()
        metrics.log_dict({"train/loss": loss})

    for step in range(FLAGS.steps):
        train_step(next(train_iter))

        if step % FLAGS.log_every == 0:
            logs = metrics.collect_group("train/")
            for k, v in logs.items():
                v = torch.stack(v).mean().item()
                logging.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    app.run(main)
