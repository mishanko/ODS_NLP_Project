import lightning as L
import torch
from torch import Tensor, nn
import torchmetrics
from torchmetrics.classification import MulticlassF1Score


class CodeClassificationModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        super().__init__()
        self.save_hyperparameters("optimizer", "scheduler", logger=False)
        self.model = model
        self.criterion = criterion
        self.metrics = nn.ModuleDict(
            {
                "train_metric": MulticlassF1Score(54, average="macro"),
                "val_metric": MulticlassF1Score(54, average="macro"),
                "test_metric": MulticlassF1Score(54, average="macro"),
            }
        )

    def configure_optimizers(
        self,
    ) -> dict[str, any]:
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/val",
                },
            }
        return {"optimizer": optimizer}

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def common_step(self, batch: tuple[Tensor, Tensor], mode: str) -> Tensor:
        assert mode in {"train", "val", "test"}
        input, target = batch
        output = self(input)
        loss: Tensor = self.criterion(output, target)
        metric: Tensor = self.metrics[f"{mode}_metric"](output, target)
        self.log(f"loss/{mode}", loss, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        self.log(f"F1Score/{mode}", metric, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor]):
        return self.common_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor]):
        return self.common_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor]):
        return self.common_step(batch, "test")
    
    def predict_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        input, target = batch
        output = self(input)
        return output.argmax(1), target
