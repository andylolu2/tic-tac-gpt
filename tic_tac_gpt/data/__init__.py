import json
from dataclasses import dataclass
from pathlib import Path

import torch
from ml_tools.preload_dataset import PreloadDataset


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

    def next_moves(self):
        if self.result != "in_progress":
            return set()
        return set(range(9)) - set(self.game_sequence)

    def next_states(self):
        for move in sorted(self.next_moves()):
            yield TicTacToeState(self.game_sequence + [move])


class TicTacToeDataset(PreloadDataset):
    bos_token: int = 12
    pad_token: int = 13
    vocab_size: int = 14
    max_seq_len: int = 11  # bos + 9 moves + result
    ENTROPY: float = 1.345254

    def __init__(self, data_file: Path | str):
        data_file = Path(data_file)
        assert data_file.suffix == ".jsonl"
        x = []
        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line)
                x.append(self.encode(["[BOS]", *item["seq"], item["result"]]))

        x = torch.nested.nested_tensor(x).to_padded_tensor(self.pad_token)
        super().__init__(x, device="cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def encode_one(cls, input: int | str) -> int:
        match input:
            case int():
                return input
            case "first":
                return 9
            case "second":
                return 10
            case "draw":
                return 11
            case "[BOS]":
                return cls.bos_token
            case "[PAD]":
                return cls.pad_token
            case _:
                raise ValueError(f"Unknown input {input}")

    @classmethod
    def decode_one(cls, input: int) -> int | str:
        match input:
            case cls.bos_token:
                return "[BOS]"
            case cls.pad_token:
                return "[PAD]"
            case 9:
                return "first"
            case 10:
                return "second"
            case 11:
                return "draw"
            case _:
                return input

    @classmethod
    def encode(cls, input: list[int | str]) -> torch.Tensor:
        return torch.tensor([cls.encode_one(c) for c in input], dtype=torch.long)

    @classmethod
    def decode(cls, input: torch.Tensor) -> list[int | str]:
        return [cls.decode_one(c) for c in input.tolist()]


if __name__ == "__main__":
    data_file = Path("out/dataset/50_50/train.jsonl")
    dataset = TicTacToeDataset(data_file)

    print(dataset[0][0])
    print(dataset.decode(dataset[0][0]))
    print(dataset.vocab_size)  # 13
