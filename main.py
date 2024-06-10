import hydra
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from sklearn.model_selection._split import train_test_split

from src import helpers as h
from src import train as tr
from src import model_builder as mb
from src import datasets as d
from src.processing import preprocess

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose('config')
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg.dataset.preprocess:
        preprocess(dataset_name=cfg.dataset.name,
                   **cfg_dict['dataset'])

    if 'inference' in cfg.mode.name:
        # // TODO: inference
        pass

    else:
        try:
            arg = sys.argv[1]

        except IndexError:
            arg = 'train'

        if 'sweep' in arg:
            # // TODO: sweep function

            wb.init(**cfg.wandb, config=cfg_dict)
            sweep_config = OmegaConf.load('conf/sweep.yaml')
            sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
            sweep_id = wb.sweep(sweep_config, project='PHASER')
            wb.agent(sweep_id, function=sweep_train, count=10)

        elif 'train' in cfg.mode.name:

            # Build RPE Dataset
            uni_cfg = cfg.dataset
            train_cfg = cfg.dataset.args.train
            val_cfg = cfg.dataset.args.val
            test_cfg = cfg.dataset.args.test

            # //TODO: training function
            splits = [cfg_dict['dataset']['args'][ds]['split'] for ds in
                      cfg_dict['dataset']['args']]

            labels = pd.read_csv(cfg.dataset.labels)
            data_len = len(labels)

            del labels

            train_idx, val_idx, test_idx = _create_data_splits(
                splits, data_len, cfg.random_seed
            )

            train_ds = d.WayneCroppedDataset(
                **uni_cfg, **train_cfg, data_idx=train_idx,
                batch_size=cfg.mode.batch_size)

            val_ds = d.WayneCroppedDataset(
                **uni_cfg, **val_cfg, data_idx=val_idx,
                batch_size=cfg.mode.batch_size)

            test_ds = d.WayneCroppedDataset(
                **uni_cfg, **test_cfg, data_idx=test_idx,
                batch_size=cfg.mode.batch_size
            )

            print(f'Input Shape: {train_ds[0][0][0].shape}')

            model = mb.build_model(cfg.model,
                                   input_shape=train_ds[0][0][0].shape,
                                   num_classes=3)

            model.summary()

            model_save_path = Path(cfg.model_save_path) / cfg.wandb.name

            wb.init(config=cfg_dict, **cfg.wandb)

            tr.train(train_ds, val_ds, model,
                     optimizer=mb._get_optimizer(
                        cfg.mode.optimizer, **cfg.mode.optimizer_args),
                     loss_fn=tf.keras.losses.CategoricalCrossentropy(),
                     train_acc_metric=tf.keras.metrics.CategoricalAccuracy(),
                     val_acc_metric=tf.keras.metrics.CategoricalAccuracy(),
                     model_save_path=model_save_path,
                     save_model=True,
                     epochs=cfg.mode.epochs)

    wb.finish()


def _create_data_splits(splits, data_len, random_seed):
    assert sum(splits) == 1

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


def sweep_train(config_defaults=None):

    cfg = config_defaults
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    splits = [cfg_dict['dataset']['args'][ds]['split'] for ds in
              cfg_dict['dataset']['args']]

    uni_cfg = cfg.dataset
    train_cfg = cfg.dataset.args.train
    val_cfg = cfg.dataset.args.val
    test_cfg = cfg.dataset.args.test

    labels = pd.read_csv(cfg.dataset.labels)
    data_len = len(labels)

    del labels

    train_idx, val_idx, test_idx = _create_data_splits(
        splits, data_len, cfg.random_seed
    )

    train_ds = d.WayneCroppedDataset(
        **uni_cfg, **train_cfg, data_idx=train_idx,
        batch_size=cfg.mode.batch_size)

    val_ds = d.WayneCroppedDataset(
        **uni_cfg, **val_cfg, data_idx=val_idx,
        batch_size=cfg.mode.batch_size)

    test_ds = d.WayneCroppedDataset(
        **uni_cfg, **test_cfg, data_idx=test_idx,
        batch_size=cfg.mode.batch_size
    )

    model = mb.build_model(cfg.model,
                           input_shape=train_ds[0][0][0].shape,
                           num_classes=3)

    tr.train(train_ds, val_ds, model,
             optimizer=mb._get_optimizer(
                 cfg.mode.optimizer, **cfg.mode.optimizer_args),
             loss_fn=tf.keras.losses.CategoricalCrossentropy(),
             train_acc_metric=tf.keras.metrics.CategoricalAccuracy(),
             val_acc_metric=tf.keras.metrics.CategoricalAccuracy(),
             epochs=cfg.mode.epochs)


if __name__ == '__main__':
    main()
