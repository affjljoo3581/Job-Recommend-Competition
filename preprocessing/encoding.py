from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.special import erfinv


def encode_numerical_data_to_gauss_rank(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    epsilon: float = 1e-4,
    noise_scale: float = 0.001,
) -> tuple[np.ndarray, Optional[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Encode the numerical data by using rank-gauss method.

    Args:
        train_data: The series (or array) of train dataset.
        test_data: The series (or array) of test dataset.
        epsilon: The epsilon which is used to clip the normalized rank to prevent
            numerical instability. Default is `1e-4`.
        noise_scale: The standard deviation of the gaussian noises which are added to
            the original data. Default is `0.001`.

    Returns:
        A tuple of normalized train and test arrays with their interpolation data.
    """
    train_noise = np.random.normal(0, 1, train_data.shape)
    train_data = train_data + noise_scale * train_data.max() * train_noise

    if test_data is not None:
        test_noise = np.random.normal(0, 1, test_data.shape)
        test_data = test_data + noise_scale * train_data.max() * test_noise

    # Calculate the train ranks and normalize them.
    train_rank = np.argsort(np.argsort(train_data))
    train_rank = 2 * train_rank / train_rank.max() - 1

    # Clip the rank scores and apply inverse gaussian.
    clipped_train_rank = np.clip(train_rank, -1 + epsilon, 1 - epsilon)
    transformed_train_data = erfinv(clipped_train_rank)

    transformed_test_data = None
    if test_data is not None:
        # Because we cannot use the entire data (including test data) to calculate the
        # rank scores of the test data, we will interpolate the scores through the train
        # data.
        order = np.argsort(train_data)
        test_rank = np.interp(test_data, train_data[order], train_rank[order])
        clipped_test_rank = np.clip(test_rank, -1 + epsilon, 1 - epsilon)
        transformed_test_data = erfinv(clipped_test_rank)

    return transformed_train_data, transformed_test_data, (train_data, train_rank)


def encode_numerical_data_to_standard_normalization(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], tuple[float, float]]:
    """Encode the numerical data by using standard normalization.

    Args:
        train_data: The series (or array) of train dataset.
        test_data: The series (or array) of test dataset.

    Returns:
        A tuple of normalized train and test arrays with their statistical informations.
    """
    # Note that we only use the statistics of train dataset.
    mean, std = train_data.mean(), train_data.std()
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    return train_data, test_data, (mean, std)


def encode_categorical_data_with_vocabulary(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    unknown_item_idx: int = 0,
) -> tuple[np.ndarray, Optional[np.ndarray], list[Any]]:
    """Encode the categorical data by using vocabulary mapping.

    Args:
        train_data: The series (or array) of train dataset.
        test_data: The series (or array) of test dataset.
        unknown_item_idx: The alternative index of unknown elements.

    Returns:
        A tuple of mapped train and test arrays with their vocabulary.
    """
    # Note that we only use the categories in train dataset.
    vocab = sorted(set(train_data))
    token_to_idx = {token: i for i, token in enumerate(vocab)}

    train_data = map(lambda x: token_to_idx.get(x, unknown_item_idx), train_data)
    train_data = np.fromiter(train_data, dtype=np.int64)

    if test_data is not None:
        test_data = map(lambda x: token_to_idx.get(x, unknown_item_idx), test_data)
        test_data = np.fromiter(test_data, dtype=np.int64)

    return train_data, test_data, vocab
