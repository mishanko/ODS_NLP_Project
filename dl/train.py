from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L
from src.pl_model import CodeClassificationModel
from src.common import create_metrics_df
import torch


def init_callbacks(config: DictConfig) -> list[L.Callback]:
    return [instantiate(x) for _, x in config.callbacks.items()]


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(config: DictConfig) -> None:
    if not config.exp_name:
        raise ValueError("`exp_name` must be set")
    tokenizer = instantiate(config.tokenizer)
    assert config.vocab_size == len(tokenizer.vocab) + 1
    dm = instantiate(config.datamodule)
    dm.setup()
    if config.random_crop:
        dm.train_dataset.tokenizer.random_crop = True

    callbacks = init_callbacks(config)
    trainer = instantiate(config.trainer)(callbacks=callbacks)
    model = instantiate(config.model)
    optimizer = instantiate(config.optim)
    scheduler = instantiate(config.scheduler)
    criterion = instantiate(config.criterion)
    if config.checkpoint:
        model = CodeClassificationModel.load_from_checkpoint(
            checkpoint_path=config.checkpoint,
            map_location="cpu",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        model = CodeClassificationModel(
            model,
            criterion,
            optimizer,
            scheduler,
        )
    trainer.fit(model, dm)

    ckpt_dir = Path(config.callbacks.model_checkpoint.dirpath)
    metrics_dir = ckpt_dir.parent / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    for ckpt in ckpt_dir.glob("*.ckpt"):
        preds = trainer.predict(model, dm.test_dataloader(), ckpt_path=ckpt)
        targets = torch.cat([x[1] for x in preds])
        preds = torch.cat([x[0] for x in preds])
        df = create_metrics_df(targets, preds, dm.train_dataset.label_encoder.classes_)
        df.to_csv(metrics_dir / f"{ckpt.name}.csv", index=False)


if __name__ == "__main__":
    train()
