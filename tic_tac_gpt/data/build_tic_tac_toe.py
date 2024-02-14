import json
from dataclasses import dataclass
from pathlib import Path

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "out/dataset", "Output directory")
flags.DEFINE_float("train_split", 0.5, "Train split")


@dataclass
class TicTacToeState:
    game_sequence: list[int]

    def __repr__(self):
        cells = [[" ", " ", " "] for _ in range(3)]
        for i, pos in enumerate(self.game_sequence):
            player = "X" if i % 2 == 0 else "O"
            cells[pos // 3][pos % 3] = player
        s = "\n".join(["|".join(row) for row in cells])
        return s

    @property
    def board(self):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i, pos in enumerate(self.game_sequence):
            player = 1 if i % 2 == 0 else -1
            board[pos // 3][pos % 3] = player
        return board

    @property
    def result(self) -> str:
        winning_positions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        first_moves = set(self.game_sequence[::2])
        second_moves = set(self.game_sequence[1::2])
        for positions in winning_positions:
            if all(pos in first_moves for pos in positions):
                return "first"
            if all(pos in second_moves for pos in positions):
                return "second"

        return "draw" if len(self.game_sequence) == 9 else "in_progress"

    def next_states(self):
        possible_moves = set(range(9)) - set(self.game_sequence)
        for move in sorted(possible_moves):
            yield TicTacToeState(self.game_sequence + [move])


def all_games():
    def _all_games_from_state(state: TicTacToeState):
        if state.result != "in_progress":
            yield state
        else:
            for next_state in state.next_states():
                yield from _all_games_from_state(next_state)

    return _all_games_from_state(TicTacToeState([]))


def main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_games = 0
    with open(out_dir / "games.jsonl", "w") as f:
        for game in all_games():
            n_games += 1
            item = {"seq": game.game_sequence, "result": game.result}
            f.write(json.dumps(item) + "\n")

    logging.info(f"Generated {n_games} games")

    with (
        open(out_dir / "games.jsonl", "r") as f,
        open(out_dir / "train.jsonl", "w") as f_train,
        open(out_dir / "test.jsonl", "w") as f_test,
    ):
        for i, line in enumerate(f):
            if i < FLAGS.train_split * n_games:
                f_train.write(line)
            else:
                f_test.write(line)


if __name__ == "__main__":
    app.run(main)
