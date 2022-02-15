import argparse
import os
import re
import warnings

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import load_train_dataset
from lightning import KNOWLightningModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def get_model_name(config: DictConfig) -> str:
    data_type = re.match(r"KNOW_(\d+).pkl", os.path.basename(config.data.filename))
    data_type = data_type.group(1)

    keywords = [
        "sid",
        f"{data_type}",
        f"{config.model.hidden_size}d",
        f"{config.model.num_transform_blocks}tb",
        f"{config.model.num_attention_blocks}ab",
        f"{config.model.num_hidden_layers}l",
        f"fold{config.data.fold_index}",
    ]
    return "-".join(keywords)


def train(config: DictConfig):
    seed_everything(config.train.random_seed)

    model_name = get_model_name(config)
    model_checkpoint = ModelCheckpoint(
        monitor="val/f1", mode="max", save_weights_only=True
    )
    model_kwargs, train_dataloader, val_dataloader = load_train_dataset(config)

    Trainer(
        gpus=1,
        logger=WandbLogger(project="job-recommend-competition", name=model_name),
        callbacks=[model_checkpoint, LearningRateMonitor()],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        gradient_clip_val=config.train.max_grad_norm,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=50,
    ).fit(KNOWLightningModule(config, model_kwargs), train_dataloader, val_dataloader)

    # After training the model, load the best-scored checkpoint and save the weights.
    model = KNOWLightningModule.load_from_checkpoint(
        model_checkpoint.best_model_path, config=config, model_kwargs=model_kwargs
    )
    state_dict = {"config": config, "state_dict": model.model.state_dict()}
    torch.save(state_dict, f"{model_name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    train(config)
