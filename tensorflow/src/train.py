import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb as wb

from omegaconf import DictConfig, ListConfig
from pathlib import Path
from sklearn.model_selection._split import train_test_split
from keras.callbacks import Callback # type: ignore
from keras.preprocessing.image import array_to_img # type: ignore
from typing import Union, Tuple, Optional

from .datasets import WayneRPEDataset
from .helpers import _get_wb_tags, _get_dataset_length, log_config_error, generate_save_path
from .model_builder import build_model

from icecream import ic

OPTIMIZERS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
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

    # Check the config
    _check_train_cfg(cfg)

    data_len = _get_dataset_length(cfg.dataset.labels)
    splits = _get_splits(cfg.dataset.args)

    # Create datasets
    train_idx, val_idx, test_idx = _create_splits(
        splits=splits,
        data_len=data_len)
    
    steps_per_epoch = len(train_idx) // cfg.mode.batch_size
    validation_steps = len(val_idx) // cfg.mode.batch_size

    # Check for leakage
    _check_leakage(train=train_idx,
                   test=test_idx,
                   val=val_idx)

    train_dataset = DATASETS.get(cfg.dataset.name.lower())(
        data_cfg=cfg.dataset,
        args=cfg.dataset.args.train,
        data_idx=train_idx,
        batch_size=cfg.mode.batch_size
    )

    val_dataset = DATASETS.get(cfg.dataset.name.lower())(
        data_cfg=cfg.dataset,
        args=cfg.dataset.args.val,
        data_idx=val_idx,
        batch_size=cfg.mode.batch_size
    )

    train_dataset_tf = _create_tf_dataset(
        train_dataset,
        in_shape=train_dataset[0][0].shape,
        out_shape=train_dataset[0][-1].shape,
        batch_size=cfg.mode.batch_size)

    val_dataset_tf = _create_tf_dataset(
        val_dataset,
        in_shape=val_dataset[0][0].shape,
        out_shape=val_dataset[0][-1].shape,
        batch_size=cfg.mode.batch_size)

    model = build_model(cfg.model,
                        input_shape=train_dataset[0][0].shape,
                        num_classes=train_dataset.n_classes)

    model_save_path = generate_save_path(
        cfg.model_save_path,
        model_name=cfg.model.name,
        file_name=cfg.wandb.run_name,
        extension='.keras',
        include_time=True
    )

    log.info(f'Saving model to: {model_save_path}')

    wb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group
    )

    metrics = list(cfg.mode.metrics)
    metric_names = list(cfg.mode.metric_names)

    optimizer_class = OPTIMIZERS.get(cfg.mode.optimizer.lower())
    optimizer_args = cfg.mode.optimizer_args.get(cfg.mode.optimizer.lower(),
                                                 {})
    optimizer = optimizer_class(**optimizer_args)

    model.compile(
        optimizer=optimizer,
        loss=cfg.mode.loss,
        metrics=metrics)

    mc_save = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor=cfg.mode.early_stopping.monitor,
        mode='min',
        save_weights_only=False,
        save_best_only=True)

    callbacks = [wb.keras.WandbMetricsLogger(), mc_save]

    if cfg.mode.early_stopping.enabled:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=cfg.mode.early_stopping.monitor,
            patience=cfg.mode.early_stopping.patience)

        callbacks.append(early_stopping)

    log.info('Training model...')

    hist = model.fit(train_dataset_tf, epochs=cfg.mode.epochs,
                     validation_data=val_dataset_tf, callbacks=callbacks,
                     steps_per_epoch=steps_per_epoch,
                     validation_steps=validation_steps)

    hist = hist.history
    best_epoch = hist['val_loss'].index(min(hist['val_loss'])) + 1

    log.info('Model Training Complete.')
    log.info(get_final_training_msg(
        hist, metrics, metric_names, best_epoch
    ))


def get_final_training_msg(history,
                           metrics: list,
                           metric_names: list,
                           epoch_n: int) -> str:

    log_txt = f"Best Epoch: {epoch_n}"
    log_txt += f"Val Loss: {history['val_loss'][epoch_n - 1]}\n"

    for m, metric in enumerate(metrics):
        log_txt += f'Val {metric_names[m]}: ' + \
            f'{history[f"val_{metric}"][epoch_n - 1]}'

    return log_txt


def _check_leakage(**kwargs: Optional[np.ndarray]):
    """
    Takes np.ndarrays for dataset indicies and finds if there is an
    intersection for any of them.
    """

    sets = {
        name: set(idx) if idx is not None else set()
        for name, idx in kwargs.items()
    }

    names = list(sets.keys())

    log.info('Checking for data leakage...')

    leaks = False
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            intersection = sets[names[i]] & sets[names[j]]

            if not intersection == set():
                leaks = True
                log.warning('Data Leakage detected in dataset indices:'
                            f'{names[i]} and {names[j]}')

    if not leaks:
        log.info('No data leakage found.')

    log.info('Finished data leakage check.')


def _get_splits(dataset_args: DictConfig) -> list:
    """
    Looks at the datasets and their arguments and gets a list of splits.
    Args:
        dataset_args (DictConfig): from cfg.dataset.args

    Returns:
        list: list of split values
    """

    return [dataset_args[arg]['split'] for arg in dataset_args]


def _create_splits(data_len: int,
                   splits: list,
                   random_seed: int = 416) -> Tuple[np.ndarray, ...]:
    """
    Creates train-test-val splits from the train configuration.
    Args:
        data_len: length of the dataset (number of samples)
        splits (list): list of splits
        random_seed (int, optional): Random seed. Defaults to 416.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple of arrays
        indicating the indices present in the train, test, val sets.
    """

    # Ensure that the splits sum to 1
    if sum(splits) != 1:
        log_config_error('Dataset splits must sum to 1.')

    idx = []
    for s in range(len(splits) - 1):
        if s == 0:
            split_range = np.arange(data_len)

        split_1 = splits[s] / sum(splits[s:])

        split_1, split_2 = train_test_split(split_range,
                                            train_size=split_1,
                                            random_state=random_seed)

        split_range = split_2

        idx.append(split_1)

    idx.append(split_2)

    return tuple(idx)


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
            '\nContinuing...\n')

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


def _dataset_generator(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        yield data


def _create_tf_dataset(dataset, in_shape, out_shape, batch_size):
    output_signature = (
        tf.TensorSpec(shape=in_shape, dtype=tf.float32),
        tf.TensorSpec(shape=out_shape)
    )

    tf_dataset = tf.data.Dataset.from_generator(
        lambda: _dataset_generator(dataset),
        output_signature=output_signature
    )

    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.map(
        lambda x, y: (x, y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    tf_dataset = tf_dataset.repeat()

    return tf_dataset
