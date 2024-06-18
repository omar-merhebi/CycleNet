import numpy as np
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import pprint
from sklearn.model_selection._split import train_test_split

from . import model_builder as mb
from . import datasets as d
from . import helpers as h
from . import processing as pr

from icecream import ic

NOW = datetime.now()
DATE = NOW.strftime('%Y-%m-%d')
TIME = NOW.strftime('%H-%M-%S')


class WandbMetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        wb.log({
            'epoch': epoch,
            'train_loss': logs.get('loss'),
            'train_acc': logs.get('categorical_accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_categorical_accuracy')})


def run_sweep():
    wb.init()
    sweep_id = wb.run.sweep_id
    
    config = OmegaConf.load(
        h.PROJECT_PATH / 'frozen_configs' / f'{sweep_id}.yaml')

    sweep_config = wb.config

    h.update_config(config, sweep_config)
    h.default_sweep_configs(config)
    train(config)


def train(config):

    if config.dataset.preprocess:
        pr.preprocess(dataset_name=config.dataset.name,
                      **config.dataset)

    train_ds, val_ds, test_ds = _load_datasets(config)

    cfg_dict = OmegaConf.to_container(config, resolve=True)
    pprint.pp(cfg_dict)

    wb.init(config=cfg_dict,
            **config.wandb)

    print(f'Train input shape: {train_ds[0][0][0].shape}')
    model = mb.build_model(config.model,
                           input_shape=train_ds[0][0][0].shape,
                           num_classes=3)

    model.compile(optimizer=mb._get_optimizer(
                        config.mode.optimizer, **config.mode.optimizer_args),
                  loss=mb._get_loss(config.mode.loss),
                  metrics=[mb._get_metric(metric)
                           for metric in config.mode.metrics])

    model.summary()

    callbacks = [WandbMetricsLogger()]

    if config.save.model:
        model_save_path = config.save.model_path
        model_save_path = Path(model_save_path) / config.wandb.name / DATE
        model_save_path.mkdir(exist_ok=True, parents=True)
        model_save_path = model_save_path / f'best_model_{TIME}.keras'
        print(f'Saving model to:\t{model_save_path}')

        mc_save = tf.keras.callbacks.ModelCheckpoint(
            str(model_save_path),
            monitor='val_loss',
            save_best_only=True
        )

        callbacks.append(mc_save)

    if config.mode.early_stopping.enabled:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.mode.early_stopping.patience
        )

        callbacks.append(early_stop)

    hist = model.fit(train_ds, epochs=config.mode.epochs,
                     validation_data=val_ds,
                     callbacks=callbacks)

    history = hist.history
    print(history)

    wb.finish()


def _load_datasets(cfg):
    # Build RPE Dataset
    uni_cfg = cfg.dataset
    train_cfg = cfg.dataset.args.train
    val_cfg = cfg.dataset.args.val
    test_cfg = cfg.dataset.args.test

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    splits = [cfg_dict['dataset']['args'][ds]['split'] for ds in
              cfg_dict['dataset']['args']]

    labels = pd.read_csv(cfg.dataset.labels)
    data_len = len(labels)

    del labels

    train_idx, val_idx, test_idx = _create_data_splits(
        splits, data_len, cfg.random_seed
    )

    cpus, gpus, total_mem = h.get_resource_allocation()
    
    if cpus >= 8:
        n_workers = (cpus - 2) // 3
        multiproc = True
        
    else:
        n_workers = 1
        multiproc = False
    
    train_ds = d.WayneCroppedDataset(
        data_idx=train_idx,
        shuffle=train_cfg.shuffle,
        balance=train_cfg.balance,
        data_dir=uni_cfg.data_dir,
        labels=uni_cfg.labels,
        channels=uni_cfg.channels,
        batch_size=cfg.mode.batch_size,
        workers=n_workers,
        use_multiprocessing=multiproc)

    val_ds = d.WayneCroppedDataset(
        data_idx=val_idx,
        shuffle=val_cfg.shuffle,
        balance=val_cfg.balance,
        data_dir=uni_cfg.data_dir,
        labels=uni_cfg.labels,
        channels=uni_cfg.channels,
        batch_size=cfg.mode.batch_size,
        workers=n_workers,
        use_multiprocessing=multiproc)

    test_ds = d.WayneCroppedDataset(
        data_idx=test_idx,
        shuffle=test_cfg.shuffle,
        balance=test_cfg.balance,
        data_dir=uni_cfg.data_dir,
        labels=uni_cfg.labels,
        channels=uni_cfg.channels,
        batch_size=cfg.mode.batch_size,
        workers=n_workers,
        use_multiprocessing=multiproc)

    return train_ds, val_ds, test_ds


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
