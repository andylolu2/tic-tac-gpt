import json
import pickle
from pathlib import Path

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
flags.DEFINE_string("output_dir", "out/model/exp1", "Output file")
flags.DEFINE_integer("batch_size", 512, "Batch size")
flags.DEFINE_integer("steps", 10000, "Steps")
flags.DEFINE_integer("log_every", 500, "Log every")
flags.DEFINE_integer("eval_every", 1000, "Eval every")


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_train = TicTacToeDataset(FLAGS.train_file)
    ds_test = TicTacToeDataset(FLAGS.test_file)
    train_loader = InfiniteDataLoader(
        ds_train, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(ds_test, batch_size=FLAGS.batch_size)

    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=1,
            d_model=128,
            d_head=64,
            n_ctx=ds_train.max_seq_len,
            d_vocab=ds_train.vocab_size,
            act_fn="solu_ln",
            attn_only=False,
            normalization_type="LN",
        ),
    )
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if "embed" in mn or "embed" in pn or pn.split(".")[-1].startswith("b"):
                no_decay.add(p)
            elif pn.split(".")[-1].lower().startswith("w"):
                decay.add(p)
            else:
                raise ValueError(f"Unknown parameter type {type(p)}")
    optimizer = AdamW(
        [
            {"params": list(decay), "weight_decay": 0.1},
            {"params": list(no_decay), "weight_decay": 0.0},
        ],
        lr=3e-4,
    )

    logging.info(f"Config:\n{json.dumps(model.cfg.to_dict(), indent=2, default=str)}")
    logging.info(f"Model has {num_params(model):,} parameters")
    with open(output_dir / "config.pkl", "wb") as f:
        pickle.dump(model.cfg.to_dict(), f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(model.cfg.to_dict(), f, indent=2, default=str)

    F = fabric.Fabric(precision="16-mixed")
    model, optimizer = F.setup(model, optimizer)
    train_loader, test_loader = F.setup_dataloaders(train_loader, test_loader)
    train_iter = iter(train_loader)

    def train_step(batch):
        (x,) = batch
        logits = model(x[:, :-1], return_type="logits")  # (b s d)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            x[:, 1:].flatten(0, 1),
            ignore_index=ds_train.pad_token,
        )
        F.backward(loss)
        optimizer.step()
        metrics.log_dict({"train/loss": loss - ds_train.ENTROPY})

    @torch.no_grad()
    def eval_step(batch):
        (x,) = batch
        logits = model(x, return_type="logits")
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            x[:, 1:].flatten(0, 1),
            ignore_index=ds_train.pad_token,
        )
        metrics.log_dict({"eval/loss": loss - ds_train.ENTROPY})

    for step in range(1, FLAGS.steps + 1):
        train_step(next(train_iter))

        if step % FLAGS.log_every == 0:
            logging.info("Step %d", step)
            logs = metrics.collect_group("train/")
            for k, v in logs.items():
                v = torch.stack(v).mean().item()
                logging.info("%s: %.4f", k, v)

        if step % FLAGS.eval_every == 0:
            logging.info("Eval at step %d", step)
            for batch in test_loader:
                eval_step(batch)
            logs = metrics.collect_group("eval/")
            for k, v in logs.items():
                v = torch.stack(v).mean().item()
                logging.info("%s: %.4f", k, v)

            F.save(output_dir / f"model_{step}.pt", model.state_dict())


if __name__ == "__main__":
    app.run(main)
