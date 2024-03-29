import json
import random
from pathlib import Path

from absl import app, flags, logging

from tic_tac_gpt.data import TicTacToeState

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "out/dataset", "Output directory")
flags.DEFINE_float("train_split", 0.5, "Train split")
flags.DEFINE_bool("random_split", False, "Random or skewed split")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_games = 0
    with open(out_dir / "games.jsonl", "w") as f:
        for game in TicTacToeState.all_games():
            n_games += 1
            item = {"seq": game.game_sequence, "result": game.result}
            f.write(json.dumps(item) + "\n")

    logging.info(f"Generated {n_games} games")

    all_indices = list(range(n_games))
    if FLAGS.random_split:
        random.shuffle(all_indices)
    train_indices = set(all_indices[: int(FLAGS.train_split * n_games)])

    with (
        open(out_dir / "games.jsonl", "r") as f,
        open(out_dir / "train.jsonl", "w") as f_train,
        open(out_dir / "test.jsonl", "w") as f_test,
    ):
        for i, line in enumerate(f):
            if i in train_indices:
                f_train.write(line)
            else:
                f_test.write(line)


if __name__ == "__main__":
    app.run(main)
