import numpy as np
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import pprint
from sklearn.model_selection._split import train_test_split
from tqdm import tqdm

from . import model_builder as mb
from . import datasets as d

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


def run_train(cfg, save_model=False):

    train_ds, val_ds, test_ds = _load_datasets(cfg)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint.pp(cfg_dict)

    wb.init(config=cfg_dict,
            **cfg.wandb)

    print(f'Train input shape: {train_ds[0][0][0].shape}')
    model = mb.build_model(cfg.model,
                           input_shape=train_ds[0][0][0].shape,
                           num_classes=3)

    model.compile(optimizer=mb._get_optimizer(
                        cfg.mode.optimizer, **cfg.mode.optimizer_args),
                  loss=mb._get_loss(cfg.mode.loss),
                  metrics=[mb._get_metric(metric)
                           for metric in cfg.mode.metrics])

    model.summary()

    callbacks = [WandbMetricsLogger()]

    if save_model:
        model_save_path = cfg.model_save_path
        model_save_path = Path(model_save_path) / cfg.wandb.name / DATE
        model_save_path.mkdir(exist_ok=True, parents=True)
        model_save_path = model_save_path / f'best_model_{TIME}.keras'
        print(f'Saving model to:\t{model_save_path}')

        mc_save = tf.keras.callbacks.ModelCheckpoint(
            str(model_save_path),
            monitor='val_loss',
            save_best_only=True
        )

        callbacks.append(mc_save)

    if cfg.mode.early_stopping.enabled:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg.mode.early_stopping.patience
        )

        callbacks.append(early_stop)

    hist = model.fit(train_ds, epochs=cfg.mode.epochs,
                     validation_data=val_ds,
                     callbacks=callbacks)

    history = hist.history

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

    return train_ds, val_ds, test_ds


def train(train_dataset, val_dataset, model, optimizer, loss_fn,
          train_acc_metric, val_acc_metric, model_save_path, save_model=False,
          epochs=10, log_step=200, val_log_step=50):

    best_val_loss = 10000

    for epoch in range(epochs):
        print(f'Start of epoch {epoch}')

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        print('Training step:')
        for step, (x_batch_train, y_batch_train, log_batch_train) in tqdm(enumerate(train_dataset),
                                                                          total=len(train_dataset)):
            if x_batch_train is None:
                break

            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optimizer, loss_fn,
                                    train_acc_metric)

            train_loss.append(float(loss_value))

        # Validation step
        print('Validation step:')
        for step, (x_batch_val, y_batch_val, log_batch_val) in tqdm(enumerate(val_dataset),
                                                                    total=len(val_dataset)):

            if x_batch_val is None:
                break

            val_loss_value = test_step(x_batch_val, y_batch_val,
                                       model, loss_fn, val_acc_metric)

            val_loss.append(float(val_loss_value))

        if np.mean(val_loss) < best_val_loss and save_model:
            best_val_loss = val_loss_value
            model.save(model_save_path)

        # Display training metrics at end of each epoch
        train_acc = train_acc_metric.result()
        print(f'Training loss over epoch: {np.mean(train_loss)}')
        print(f'Training acc over epoch: {float(train_acc):.4f}')

        val_acc = val_acc_metric.result()
        print(f'Validation loss over epoch: {np.mean(val_loss)}')
        print(f'Validation acc over epoch: {float(val_acc):.4f}')

        # Reset metrics
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        wb.log({'epochs': epoch,
                'loss': np.mean(train_loss),
                'acc': float(train_acc),
                'val_loss': np.mean(val_loss),
                'val_acc': float(val_acc)})


def train_step(x, y, model, optimizer,
               loss_fn, train_acc_metric):

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_val = loss_fn(y, logits)

    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_val


def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_val


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
