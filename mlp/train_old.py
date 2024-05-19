import lightning as L
import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassF1Score
import json
from pathlib import Path
import sentencepiece as spm
from datamodule import CodeClassificationDatamodule
from pl_model import CodeClassificationModel
from functools import partial
from lightning.pytorch import callbacks


def open_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def init_model(tokenizer):
    optim = partial(
        torch.optim.AdamW,
        lr=0.001,
        weight_decay=0.0,
    )
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.5, patience=20, min_lr=5e-5)
    model = nn.Sequential(
        nn.Embedding(tokenizer.GetPieceSize(), 32),
        # nn.Flatten(),
        nn.Linear(32, 256),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(1024, 54),
    )
    criterion = nn.CrossEntropyLoss()
    model = CodeClassificationModel(model=model, criterion=criterion, optimizer=optim, scheduler=scheduler)
    return model


if __name__ == "__main__":
    datadir = "/data/guesslang_data/Dataset/Data"
    label_mapper = open_json("label_mapper.json")
    tokenizer = spm.SentencePieceProcessor(model_file="../tokenizers/tok.model", num_threads=1)
    dm = CodeClassificationDatamodule(
        datadir,
        label_mapper,
        tokenizer,
        context_len=1024,
        random_crop=True,
        num_workers=4,
        batch_size=2048,
    )
    dm.setup()
    model = init_model(tokenizer)

    trainer = L.Trainer(
        num_sanity_val_steps=10,
        gradient_clip_val=3.0,
        max_epochs=500,
        accelerator="gpu",
        precision="16-mixed",
        devices=[7],
        callbacks=[
            callbacks.RichProgressBar(),
            callbacks.ModelCheckpoint("models", monitor="loss/val", save_weights_only=True, save_top_k=2),
        ],
    )

    trainer.fit(model, datamodule=dm)
