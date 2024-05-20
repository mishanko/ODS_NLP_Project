import pickle
from functools import partial
from pathlib import Path

import lightning as L
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .dataset import CodeClassificationDataset
from .tokenizer import RegexTokenizer


class CodeClassificationDatamodule(L.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        label_encoder: str | LabelEncoder,
        tokenizer: RegexTokenizer,
        max_tokens: int = 128,
        random_crop: bool = False,
        num_workers: int = 0,
        batch_size: int = 128,
    ):
        super().__init__()
        if isinstance(label_encoder, (str, Path)):
            with open(label_encoder, "rb") as f:
                label_encoder = pickle.load(f)
        self._default_loader = partial(
            DataLoader,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        self._default_dataset = partial(
            CodeClassificationDataset,
            datadir=datadir,
            label_encoder=label_encoder,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            random_crop=random_crop,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._default_dataset(mode="train")
            self.val_dataset = self._default_dataset(mode="valid")
        if stage == "test" or stage is None:
            self.test_dataset = self._default_dataset(mode="test")

    def train_dataloader(self):
        return self._default_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._default_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._default_loader(self.test_dataset, shuffle=False)
