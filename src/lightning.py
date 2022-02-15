from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from modeling import SiDClassifier, SiDConfig


class KNOWLightningModule(LightningModule):
    def __init__(self, config: DictConfig, model_kwargs: dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = SiDClassifier(SiDConfig(**model_kwargs, **config.model))
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch: tuple[torch.Tensor, ...], idx: int) -> torch.Tensor:
        logits = self.model(*batch[:3])
        loss = self.criterion(logits, batch[3])
        acc = (logits.argmax(-1) == batch[3]).float().mean()

        self.log_dict({"train/loss": loss, "train/acc": acc})
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(*batch[:3]), batch[3]

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        logits = torch.cat([logits for logits, _ in outputs])
        labels = torch.cat([labels for _, labels in outputs])

        loss = self.criterion(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        f1 = f1_score(labels.tolist(), logits.argmax(-1).tolist(), average="macro")

        self.log_dict({"val/loss": loss, "val/acc": acc, "val/f1": f1})

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.get_parameter_groups(), **self.config.optim)
        scheduler = CosineAnnealingLR(
            optimizer, self.config.train.epochs, self.config.optim.lr * 1e-3
        )
        return [optimizer], [{"scheduler": scheduler}]

    def get_parameter_groups(self) -> list[dict[str, Any]]:
        do_decay, no_decay = [], []
        for param in self.parameters():
            if param.ndim < 2:
                no_decay.append(param)
            else:
                do_decay.append(param)
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]
