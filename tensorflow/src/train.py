import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb as wb

from omegaconf import DictConfig, ListConfig
from pathlib import Path
from sklearn.model_selection._split import train_test_split
from typing import Union, Tuple

from .datasets import WayneRPEDataset
from .helpers import _get_wb_tags, log_config_error, generate_save_path
from .model_builder import build_model

from icecream import ic

AUTOTUNE = tf.data.AUTOTUNE

OPTIMIZERS = {
    'adam': tf.keras.optimizers.Adam
}

DATASETS = {
    'wayne_rpe': WayneRPEDataset
}

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    """
    Run model training.
    Args:
        cfg (DictConfig): The full hydra config.
    """

    # // TODO Finish train

    # Check the config
    _check_train_cfg(cfg)

    # Create datasets
    train_idx, test_idx, val_idx = _create_splits(
        cfg.dataset.labels,
        splits=cfg.dataset.splits)

    train_dataset = DATASETS.get(cfg.dataset.name)(
        cfg, train_idx, augment=cfg.dataset.augment,
        fill=cfg.dataset.fill,
        mask=cfg.dataset.mask
    )

    val_dataset = DATASETS.get(cfg.dataset.name)(cfg, val_idx)

    test_dataset = DATASETS.get(cfg.dataset.name)(cfg, test_idx)

    # Init Wandb
    # run = wb.init(
    #     project=cfg.wandb.project,
    #     group=cfg.wandb.group,
    #     name=cfg.wandb.run_name,
    #     tags=_get_wb_tags(cfg)
    # )

    model_save_path = generate_save_path(
        cfg.model_save_path,
        model_name=cfg.model.name,
        file_name=cfg.wandb.run_name,
        extension='.tf',
        include_time=True
    )

    ic(model_save_path)

    # Build model
    # model = build_model(cfg, input_shape=# //TODO,
    #                     num_classes=#//TODO
    #                     )

    # metrics = cfg.mode.metrics
    # metric_names = cfg.mode.metric_names


    # optimizer = OPTIMIZERS.get(cfg.mode.optimizer.lower())(
    #     learning_rate=cfg.mode.learning_rate
    # )

    # model.compile(
    #     optimizer=optimizer,
    #     loss=cfg.mode.loss,
    #     metrics=cfg.mode.metrics
    #     )

    run.finish()


def _create_splits(labels_csv: Union[str, Path],
                   splits: DictConfig,
                   random_seed: int = 416) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Creates train-test-val splits from the train configuration.
    Args:
        labels_csv (Union[str, Path]): path to the labels csv
        splits (dict): splits configuration
        random_seed (int, optional): Random seed. Defaults to 416.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple of arrays
        indicating the indices present in the train, test, val sets.
    """

    labels_csv = pd.read_csv(labels_csv)

    train_idx, test_idx = train_test_split(np.arange(len(labels_csv)),
                                           train_size=splits.train,
                                           random_state=random_seed)

    test_idx, val_idx = train_test_split(
        test_idx,
        train_size=splits.test / (splits.val + splits.test),
        random_state=random_seed)

    return train_idx, test_idx, val_idx


def _check_train_cfg(cfg: DictConfig) -> None:
    """
    Validates the training configuration
    Args:
        cfg (DictConfig): the input configuration
    """

    metrics = cfg.mode.metrics
    metric_names = cfg.mode.metric_names

    if type(metrics) is not ListConfig or type(metric_names) is not ListConfig:
        log.error(
            'Metrics and Metric names should be of type list, got:'
            f'type(metrics):\t{type(metrics)}'
            f'type(metric_names):\t{type(metric_names)}'
            '\n\Continuing...\n')

    if len(metrics) != len(metric_names):
        log_config_error(
            'Metric types and names must match, got:'
            f'N Metrics:\t{len(metrics)}'
            f'N Metric Names:\t{len(metric_names)}'
        )

    optimizer = cfg.mode.optimizer.lower()

    if optimizer not in OPTIMIZERS:
        log_config_error(
            f'Invalid optimizer name: {optimizer} ,'
            f'must be one of: {OPTIMIZERS.keys}'
        )
