import numpy as np
import tensorflow as tf
import wandb as wb

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

NOW = datetime.now()
DATE = NOW.strftime('%Y-%m-%d')
TIME = NOW.strftime('%H-%M-%S')


def train(train_dataset, val_dataset, model, optimizer, loss_fn,
          train_acc_metric, val_acc_metric, model_save_path, save_model=False,
          epochs=10, log_step=200, val_log_step=50):

    best_val_loss = 10000

    if save_model:
        model_save_path = Path(model_save_path) / DATE
        model_save_path.mkdir(exist_ok=True, parents=True)
        model_save_path = model_save_path / f'best_model_{TIME}.keras'
        print(model_save_path)

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
