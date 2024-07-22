import numpy as np
import pandas as pd
import pprint
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection._split import train_test_split
from tqdm import tqdm

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
    # //TODO Move this to main
    wb.init()
    sweep_id = wb.run.sweep_id

    config = OmegaConf.load(
        h.PROJECT_PATH / 'frozen_configs' / f'{sweep_id}.yaml')

    sweep_config = wb.config

    h.update_config(config, sweep_config)
    h.default_sweep_configs(config)
    setup_training(config)


def setup_training(config):

    if config.dataset.preprocess:
        pr.preprocess(dataset_name=config.dataset.name,
                      **config.dataset)

    train_ds, val_ds, test_ds = _load_datasets(config)

    cfg_dict = OmegaConf.to_container(config, resolve=True)
    pprint.pp(cfg_dict)

    # //TODO: move this to main
    wb.init(config=cfg_dict,
            **config.wandb)

    model = mb.build_model(config.model,
                           input_shape=train_ds[0][0][0].shape,
                           num_classes=3)

    optim = mb._get_optimizer(config.mode.optimizer,
                              **config.mode.optimizer_args)
    loss_fn = mb._get_loss(config.mode.loss)

    train_acc_metric = mb._get_metric(config.mode.metrics)
    val_acc_metric = mb._get_metric(config.mode.metrics)

    if model and not _check_zero_dim_layers(model):
        print('Model Summary:')
        print(model.summary())
        train(train_ds, val_ds, model, optim, train_acc_metric, val_acc_metric,
              loss_fn, config.mode.epochs)

    else:
        wb.log({'epoch': 1,
                'loss': 1.1,
                'accuracy': 0,
                'val_loss': 1.1,
                'val_accuracy': 0})


def train(train_data, val_data, model, optim, train_acc_metric, val_acc_metric,
          loss_fn, epochs, log_step=200, val_log_step=50):

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / {epochs}")

        train_loss = []
        val_loss = []

        # Iterate over batches
        print(f'Number of batches: {len(train_data)}')
        for step, (x_batch_train, y_batch_train) in tqdm(
                enumerate(train_data), total=len(train_data)):

            # This is needed to ensure the logic does not go past final batch.
            # Not sure why this behavior happens.
            if x_batch_train.shape[0] == 0:
                break

            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optim, loss_fn, train_acc_metric)

            train_loss.append(loss_value)

        for step, (x_batch_val, y_batch_val) in enumerate(val_data):
            if x_batch_val.shape[0] == 0:
                break

            val_loss_value = test_step(x_batch_val, y_batch_val,
                                       model, loss_fn, val_acc_metric)

            val_loss.append(val_loss_value)

        # Display metrics
        train_acc = train_acc_metric.result()
        print(f"Training accuracy over epoch: {float(train_acc):.4f}")

        val_acc = val_acc_metric.result()
        print(f'Validation accuracy after epoch: {float(val_acc):.4f}')

        # Reset accuracies
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        wb.log({'epoch': epoch + 1,
                'loss': np.mean(train_loss),
                'accuracy': float(train_acc),
                'val_loss': np.mean(val_loss),
                'val_accuracy': float(val_acc)})


def train_step(x, y, model, optim, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_val = loss_fn(y, logits)

    grads = tape.gradient(loss_val, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_val


def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_val


def _check_zero_dim_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        else:
            if 0 in layer.input.shape:
                print(f'Layer {layer} has zero dim. Aborting...')
                return True

    return False


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

    # Check for data leakage
    inter_tr_va = set(train_idx).intersection(set(val_idx))
    inter_tr_ts = set(train_idx).intersection(set(test_idx))
    inter_va_ts = set(val_idx).intersection(set(test_idx))

    all_inter = inter_tr_va.union(inter_tr_ts).union(inter_va_ts)

    cpus, gpus, total_mem = h.get_resource_allocation()

    if all_inter:
        print(f'WARNING: Data leakage has been found, indices: {all_inter}')

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
