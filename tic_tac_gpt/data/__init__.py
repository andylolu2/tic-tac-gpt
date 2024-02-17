import json
from pathlib import Path

import torch
from ml_tools.preload_dataset import PreloadDataset


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
                x.append(self.encode(item["seq"] + [item["result"]]))

        x = torch.stack(x)
        super().__init__(x, device="cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def encode(cls, input: list[int | str]) -> torch.Tensor:
        x = [cls.bos_token]
        for c in input:
            match c:
                case int():
                    x.append(c)
                case "first":
                    x.append(9)
                case "second":
                    x.append(10)
                case "draw":
                    x.append(11)
        x = x + [cls.pad_token] * (cls.max_seq_len - len(x))
        return torch.tensor(x, dtype=torch.long)

    @classmethod
    def decode(cls, input: torch.Tensor) -> list[int | str]:
        x = []
        for c in input:
            match c.item():
                case cls.bos_token:
                    x.append("[BOS]")
                case cls.pad_token:
                    pass
                case 9:
                    x.append("first")
                case 10:
                    x.append("second")
                case 11:
                    x.append("draw")
                case _:
                    x.append(c.item())
        return x


if __name__ == "__main__":
    data_file = Path("out/dataset/50_50/train.jsonl")
    dataset = TicTacToeDataset(data_file)

    print(dataset[0][0])
    print(dataset.decode(dataset[0][0]))
    print(dataset.vocab_size)  # 13
