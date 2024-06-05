import logging
import tensorflow as tf
import wandb as wb

from omegaconf import DictConfig

from .datasets import WayneRPEDataset
from .helpers import ConfigError, _get_wb_tags
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


def run(cfg: DictConfig) -> None:
    """
    Detects which run mode to use and runs it
    Args:
        cfg (DictConfig): The full hydra config
    """

    mode = cfg.mode.name.lower()

    if mode == 'train':
        model = build_model(cfg.model)
        train(cfg, model)


def train(cfg: DictConfig, model: tf.keras.Model) -> None:

    # Init Wandb
    run = wb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.run_name,
        tags=_get_wb_tags(cfg)
    )

    metrics = cfg.mode.metrics
    metric_names = cfg.mode.metric_names

    if type(metrics) is not list or type(metric_names) is not list:
        log.critical(
            'Metrics and Metric names must be of type list, got:'
            f'type(metrics):\t{type(metrics)}'
            f'type(metric_names):\t{type(metric_names)}'
            )

        raise ConfigError(
            'Metrics and Metric names must be of type list, got:'
            f'type(metrics):\t{type(metrics)}'
            f'type(metric_names):\t{type(metric_names)}'
        )

    if len(metrics) != len(metric_names):
        log.critical(
            'Metric types and names must match, got:'
            f'N Metrics:\t{len(metrics)}'
            f'N Metric Names:\t{len(metric_names)}'
        )
        
        raise ConfigError(
            'Metric types and names must match, got:'
            f'N Metrics:\t{len(metrics)}'
            f'N Metric Names:\t{len(metric_names)}'
        )

    optimizer = OPTIMIZERS.get(cfg.mode.optimizer.lower())(
        learning_rate=cfg.mode.learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=cfg.mode.loss,
        metrics=cfg.mode.metrics
        )

    run.finish()
