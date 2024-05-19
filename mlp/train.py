from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L
from pl_model import CodeClassificationModel
from common import create_metrics_df
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
    
    callbacks = init_callbacks(config)
    trainer = instantiate(config.trainer)(callbacks=callbacks)
    model = instantiate(config.model)
    optimizer = instantiate(config.optim)
    scheduler = instantiate(config.scheduler)
    criterion = instantiate(config.criterion)
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
        

    # X_test, y_test = read_dataset(list(datadirs["test"].glob("*.*")), config.max_workers)
    # y_test = le.transform(y_test)
    # preds = pipeline.predict(X_test)
    # df = create_metrics_df(y_test, preds, le.classes_)
    # df.to_csv(outputdir / "test_metrics.csv", index=False)


if __name__ == "__main__":
    train()
