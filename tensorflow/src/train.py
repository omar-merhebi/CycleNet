import logging
import numpy as np
import tensorflow as tf
import wandb as wb

from omegaconf import DictConfig
from pathlib import Path
from typing import Union, Tuple

from .datasets import WayneRPEDataset
from .helpers import _get_wb_tags, _log_config_error
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
    # // TODO Finish train
    # Check the config
    _check_train_cfg(cfg)

    # Create datasets

    # Init Wandb
    run = wb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.run_name,
        tags=_get_wb_tags(cfg)
    )

    # Build model
    model = build_model(cfg, input_shape=# //TODO,
                        num_classes=#//TODO
                        )
    
    metrics = cfg.mode.metrics
    metric_names = cfg.mode.metric_names


    optimizer = OPTIMIZERS.get(cfg.mode.optimizer.lower())(
        learning_rate=cfg.mode.learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=cfg.mode.loss,
        metrics=cfg.mode.metrics
        )

    run.finish()


def _create_splits(labels: Union[str, Path],
                   splits: dict,
                   random_seed: int = 416) -> Tuple[np.ndarray]:
    # // TODO: create splits
    pass


def _check_train_cfg(cfg: DictConfig) -> None:

    metrics = cfg.mode.metrics
    metric_names = cfg.mode.metric_names

    if type(metrics) is not list or type(metric_names) is not list:
        log.error(
            'Metrics and Metric names should be of type list, got:'
            f'type(metrics):\t{type(metrics)}'
            f'type(metric_names):\t{type(metric_names)}'
            '\n\Continuing...\n')

    if len(metrics) != len(metric_names):
        _log_config_error(
            'Metric types and names must match, got:'
            f'N Metrics:\t{len(metrics)}'
            f'N Metric Names:\t{len(metric_names)}'
        )

    optimizer = cfg.mode.optimizer.lower()

    if optimizer not in OPTIMIZERS:
        _log_config_error(
            f'Invalid optimizer name: {optimizer} ,'
            f'must be one of: {OPTIMIZERS.keys}'
        )
        