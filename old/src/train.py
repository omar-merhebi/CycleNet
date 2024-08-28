import numpy as np
import pandas as pd
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path
from pprint import pp
from sklearn.model_selection._split import train_test_split
from tqdm import tqdm

from . import model_builder as mb
from . import datasets as d
from . import helpers as h
from . import processing as pr

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
    setup_training(config, finish=False)


def setup_training(config, finish=True):
    config_dict = OmegaConf.to_container(config, resolve=True)
    wb.config.update(config_dict)

    if config.dataset.preprocess:
        pr.preprocess(dataset_name=config.dataset.name,
                      **config.dataset)

    train_ds, val_ds, test_ds = _load_datasets(config)

    model = mb.build_model(config.model,
                           input_shape=train_ds[0][0][0].shape,
                           num_classes=3)

    optim = mb._get_optimizer(config.mode.optimizer,
                              **config.mode.optimizer_args)
    loss_fn = mb._get_loss(config.mode.loss)

    # // TODO: move all training params to .model
    train_metric = mb._get_metric(config.mode.metrics)
    val_metric = mb._get_metric(config.mode.metrics)
    test_metric = mb._get_metric(config.mode.metrics)

    if model and not _check_zero_dim_layers(model):
        print('Run Config:')
        pp(config_dict)
        print('Model Summary:')
        print(model.summary())
        train(config, train_ds, val_ds, test_ds, model, optim,
              train_metric, val_metric, test_metric,
              loss_fn, config.mode.epochs)

    if finish:
        wb.finish()


def train(config, train_data, val_data, test_data, model, optim,
          train_metric, val_metric, test_metric,
          loss_fn, epochs):

    early_stopping = config.mode.early_stopping.enabled
    min_epochs = config.mode.early_stopping.min_epochs
    epochs_before_stop = config.mode.early_stopping.patience
    best_val_loss = np.inf

    if config.mode.save_model.enabled:
        model_save_dir = Path(config.mode.save_model.path)
        model_save_dir /= config.wandb.name
        model_save_dir /= DATE
        model_save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if epoch + 1 >= min_epochs and epochs_before_stop <= 0 \
                and early_stopping:
            print('Reached stopping patience')
            break

        print(f"\nEpoch {epoch + 1} / {epochs}")

        train_loss = []
        val_loss = []
        test_loss = []

        # Iterate over batches
        for step, (x_batch_train, y_batch_train, meta) in tqdm(
                enumerate(train_data), total=len(train_data)):

            # This is needed to ensure the logic does not go past final batch.
            # Not sure why this behavior happens.
            if x_batch_train.shape[0] == 0:
                break

            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optim, loss_fn, train_metric)

            train_loss.append(loss_value)

        for step, (x_batch_val, y_batch_val, meta) in enumerate(val_data):
            if x_batch_val.shape[0] == 0:
                break

            val_loss_value = val_step(x_batch_val, y_batch_val,
                                      model, loss_fn,
                                      val_metric)

            val_loss.append(val_loss_value)

        # Display metrics
        train_acc = train_metric.result()
        print(f"Training accuracy over epoch: {float(train_acc):.4f}")

        val_acc = val_metric.result()
        print(f'Validation accuracy after epoch: {float(val_acc):.4f}')

        # Reset accuracies
        train_metric.reset_state()
        val_metric.reset_state()

        m_train_loss = np.mean(train_loss)
        m_val_loss = np.mean(val_loss)

        wb.log({'epoch': epoch + 1,
                'loss': m_train_loss,
                'accuracy': float(train_acc),
                'val_loss': m_val_loss,
                'val_accuracy': float(val_acc)})

        if m_val_loss < best_val_loss:
            best_val_loss = m_val_loss
            wb.summary['best_val_loss'] = best_val_loss

            epochs_before_stop = config.mode.early_stopping.patience

            model_save_path = model_save_dir / \
                (f'epoch_{epoch}' + f'_vloss_{best_val_loss:.2f}.h5')
            if config.mode.save_model.enabled:
                model.save(model_save_path)

        else:
            epochs_before_stop -= 1

    # Now run on test dataset:
    colnames = ['sample_id', 'true_label', 'prediction', 'confidence']
    results_df = pd.DataFrame(columns=colnames)

    for step, (x_batch_test, y_batch_test, meta) in enumerate(test_data):
        if x_batch_test.shape[0] == 0:
            break
        test_loss_value, partial_results_df = test_step(x=x_batch_test,
                                                        y=y_batch_test,
                                                        meta=meta,
                                                        model=model,
                                                        class_labels=test_data.classes,
                                                        metric=test_metric,
                                                        loss_fn=loss_fn,
                                                        df_colnames=colnames)

        results_df = pd.concat([results_df, partial_results_df],
                               ignore_index=True)

        test_loss.append(test_loss_value)

    m_test_loss = np.mean(test_loss)

    test_acc = test_metric.result()
    print(f'Test Accuracy:\t{test_acc}')

    wb.summary['test_loss'] = m_test_loss
    wb.summary['test_accuracy'] = float(test_acc)

    if config.mode.save_test_results.enabled:
        save_results = Path(config.mode.save_test_results.path)
        save_results /= config.wandb.name
        save_results /= DATE

        save_results.mkdir(parents=True, exist_ok=True)

        save_results /= 'test_data_results.csv'

        results_df.to_csv(save_results)


def test_step(x, y, meta, model, class_labels, metric, loss_fn, df_colnames,
              sample_id_key='cell_id'):

    test_logits = model.predict(x, verbose=0)
    loss_val = loss_fn(y, test_logits)
    metric.update_state(y, test_logits)

    confidences = np.max(test_logits, axis=-1)
    predictions = np.argmax(test_logits, axis=-1)
    predictions = np.array([class_labels[i] for i in predictions])

    true_lab = y.numpy()
    true_lab = np.argmax(true_lab, axis=-1)
    true_lab = np.array([class_labels[i] for i in true_lab])

    ids = meta[sample_id_key]

    stacked = np.vstack([ids, true_lab, predictions, confidences])

    temp_df = pd.DataFrame(stacked.T, columns=df_colnames)

    return loss_val, temp_df


def train_step(x, y, model, optim, loss_fn, metric):
    with tf.GradientTape() as tape:
        train_logits = model(x, training=True)
        loss_val = loss_fn(y, train_logits)

    grads = tape.gradient(loss_val, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))

    metric.update_state(y, train_logits)

    return loss_val


def val_step(x, y, model, loss_fn, metric):
    val_logits = model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    metric.update_state(y, val_logits)

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
        batch_size=1,
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
