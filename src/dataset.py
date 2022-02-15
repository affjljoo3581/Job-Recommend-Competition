import pickle
from typing import Any

import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset


def load_train_dataset(
    config: DictConfig,
) -> tuple[dict[str, Any], DataLoader, DataLoader]:
    with open(config.data.filename, "rb") as fp:
        dataset = pickle.load(fp)

    model_kwargs = {
        "num_categories_list": list(map(len, dataset["categories_list"])),
        "num_numerical_features": len(dataset["numerical_columns"]),
        "num_text_features": len(dataset["text_columns"]),
        "text_input_size": dataset["text_embeddings"].shape[2],
        "num_labels": len(dataset["label_list"]),
    }

    # Convert the arrays to the pytorch tensors.
    categorical_inputs = torch.from_numpy(dataset["categorical"])
    numerical_inputs = torch.from_numpy(dataset["numerical"])
    text_inputs = torch.from_numpy(dataset["text_embeddings"])
    labels = torch.from_numpy(dataset["labels"])

    # Use a stratified k-fold strategy to split the data into train and validation sets.
    kfold = StratifiedKFold(config.data.num_folds, shuffle=True, random_state=42)
    splits = list(kfold.split(dataset["labels"], dataset["labels"]))
    train_indices, val_indices = splits[config.data.fold_index]

    # Wrap the dataset tensors to the `TensorDataset` class. Note that we will move all
    # tensors to GPU memory to reduce the bottleneck in every batches.
    wrapped_dataset = TensorDataset(
        categorical_inputs.cuda(),
        numerical_inputs.cuda(),
        text_inputs.cuda(),
        labels.cuda(),
    )

    # Create dataloaders for train and validation.
    train_dataloader = DataLoader(
        Subset(wrapped_dataset, train_indices),
        batch_size=config.train.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        Subset(wrapped_dataset, val_indices), batch_size=config.train.batch_size
    )
    return model_kwargs, train_dataloader, val_dataloader


def load_test_dataset(
    dataset_filename: str, batch_size: int = 256
) -> tuple[dict[str, Any], list[int], list[int], DataLoader]:
    with open(dataset_filename, "rb") as fp:
        dataset = pickle.load(fp)

    model_kwargs = {
        "num_categories_list": list(map(len, dataset["categories_list"])),
        "num_numerical_features": len(dataset["numerical_columns"]),
        "num_text_features": len(dataset["text_columns"]),
        "text_input_size": dataset["text_embeddings"].shape[-1],
        "num_labels": len(dataset["label_list"]),
    }

    # Wrap the dataset tensors to the `TensorDataset` class. Note that we will move all
    # tensors to GPU memory to reduce the bottleneck in every batches.
    wrapped_dataset = TensorDataset(
        torch.from_numpy(dataset["categorical"]).cuda(),
        torch.from_numpy(dataset["numerical"]).cuda(),
        torch.from_numpy(dataset["text_embeddings"]).cuda(),
    )

    # Create a dataloader for test dataset.
    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size)
    return model_kwargs, dataset["index"], dataset["label_list"], dataloader
