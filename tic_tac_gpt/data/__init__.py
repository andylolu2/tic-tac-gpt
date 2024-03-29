import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from ml_tools.preload_dataset import PreloadDataset


@dataclass
class TicTacToeState:
    game_sequence: list[int]

    def __repr__(self):
        def board_cells(seq):
            cells = [[" ", " ", " "] for _ in range(3)]
            for i, pos in enumerate(seq):
                player = "X" if i % 2 == 0 else "O"
                cells[pos // 3][pos % 3] = player
            rows = ["|".join(row) for row in cells]
            return rows

        boards = []
        for i in range(len(self.game_sequence) + 1):
            boards.append(board_cells(self.game_sequence[:i]))

        full_rows = []
        for row in zip(*boards):
            full_rows.append("  ".join(row))
        full_game = "\n".join(full_rows)
        return full_game

    def __len__(self):
        return len(self.game_sequence)

    @staticmethod
    def all_games() -> Iterator["TicTacToeState"]:
        def _all_games_from_state(state: TicTacToeState):
            is_terminal = True
            for next_state in state.next_states():
                yield from _all_games_from_state(next_state)
                is_terminal = False

            if is_terminal:
                yield state

        return _all_games_from_state(TicTacToeState([]))

    @property
    def board(self):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i, pos in enumerate(self.game_sequence):
            player = 1 if i % 2 == 0 else 2
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
                return "[X]"
            if all(pos in second_moves for pos in positions):
                return "[O]"

        return "[D]" if len(self.game_sequence) == 9 else "in_progress"

    def next_moves(self):
        if self.result != "in_progress":
            return set()
        return set(range(9)) - set(self.game_sequence)

    def next_states(self):
        for move in sorted(self.next_moves()):
            yield TicTacToeState(self.game_sequence + [move])

    def sequence(self):
        for i in range(len(self.game_sequence) + 1):
            yield TicTacToeState(self.game_sequence[:i])


class TicTacToeDataset(PreloadDataset):
    bos_token: int = 12
    pad_token: int = 13
    vocab_size: int = 14
    max_seq_len: int = 11  # bos + 9 moves + result
    ENTROPY: float = 1.345254

    @staticmethod
    def from_file(data_file: Path | str) -> "TicTacToeDataset":
        data_file = Path(data_file)
        assert data_file.suffix == ".jsonl"
        x = []
        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line)
                x.append(TicTacToeDataset.encode(["[B]", *item["seq"], item["result"]]))
        x = torch.nested.nested_tensor(x).to_padded_tensor(TicTacToeDataset.pad_token)
        return TicTacToeDataset(x)

    @classmethod
    def from_states(cls, states: Iterator[TicTacToeState]) -> "TicTacToeDataset":
        x = []
        for state in states:
            x.append(cls.state_to_tensor(state))
        x = torch.nested.nested_tensor(x).to_padded_tensor(TicTacToeDataset.pad_token)
        return TicTacToeDataset(x)

    @classmethod
    def state_to_tensor(cls, state: TicTacToeState) -> torch.Tensor:
        seq = ["[B]", *state.game_sequence]
        result = state.result
        if result != "in_progress":
            seq.append(result)
        return cls.encode(seq)

    @classmethod
    def tensor_to_state(cls, tensor: torch.Tensor) -> TicTacToeState:
        seq = cls.decode(tensor)
        seq = [s for s in seq if isinstance(s, int)]
        return TicTacToeState(seq)

    @classmethod
    def encode_one(cls, input: int | str) -> int:
        match input:
            case int():
                return input
            case "[X]":
                return 9
            case "[O]":
                return 10
            case "[D]":
                return 11
            case "[B]":
                return cls.bos_token
            case "[P]":
                return cls.pad_token
            case _:
                raise ValueError(f"Unknown input {input}")

    @classmethod
    def decode_one(cls, input: int) -> int | str:
        match input:
            case cls.bos_token:
                return "[B]"
            case cls.pad_token:
                return "[P]"
            case 9:
                return "[X]"
            case 10:
                return "[O]"
            case 11:
                return "[D]"
            case _:
                return input

    @classmethod
    def encode(cls, input: list[int | str]) -> torch.Tensor:
        return torch.tensor([cls.encode_one(c) for c in input], dtype=torch.long)

    @classmethod
    def decode(cls, input: torch.Tensor) -> list[int | str]:
        return [cls.decode_one(c) for c in input.tolist()]


def tensor_to_state(tensor: torch.Tensor) -> TicTacToeState:
    seq = TicTacToeDataset.decode(tensor)
    seq = [s for s in seq if isinstance(s, int)]
    return TicTacToeState(seq)


if __name__ == "__main__":
    data_file = Path("out/dataset/50_50/train.jsonl")
    dataset = TicTacToeDataset.from_file(data_file)

    print(dataset[0][0])
    print(dataset.decode(dataset[0][0]))
    print(dataset.vocab_size)  # 13
