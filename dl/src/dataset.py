import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset

from .common import read_file
from .tokenizer import RegexTokenizer


class CodeClassificationDataset(Dataset):
    max_workers: int = 10

    def __init__(
        self,
        datadir: str,
        mode: str,
        label_encoder: LabelEncoder,
        tokenizer: RegexTokenizer,
        max_tokens: int = 128,
        random_crop: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(label_encoder, (str, Path)):
            with open(label_encoder, "rb") as f:
                label_encoder = pickle.load(f)
        datadir = Path(datadir)
        self.datadir = datadir / mode
        assert self.datadir.exists()
        files: list[tuple[Path, int]] = [
            (x, x.suffix) for x in self.datadir.glob("*.*")
        ]
        self.labels = label_encoder.transform([x[1] for x in files])
        self.label_encoder = label_encoder
        self.files = [x[0] for x in files]
        self.tokenizer = tokenizer
        self.max_tokoens = max_tokens
        self.random_crop = random_crop
        self.mode = mode

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        text = read_file(self.files[idx])[0]
        label = self.labels[idx]
        tokens = torch.tensor(self.tokenizer(text))
        return tokens, torch.tensor(label)
