import logging
import numpy as np
import tensorflow as tf
import wandb as wb

log = logging.getLogger(__name__)

OPTIMIZERS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
}


def train(train_dataset, val_dataset, model, optimizer, loss_fn,
          train_acc_metric, val_acc_metric, epochs=10, log_step=200,
          val_log_step=50):

    for epoch in range(epochs):
        log.info(f'Start of epoch {epoch}')

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optimizer, loss_fn, 
                                    train_acc_metric)

            train_loss.append(float(loss_value))

        # Validation step
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val,
                                       model, loss_fn, val_acc_metric)

            val_loss.append(float(val_loss_value))

        # Display training metrics at end of each epoch
        train_acc = train_acc_metric.result()
        log.info(f'Training acc over epoch: {float(train_acc):.4f}')

        val_acc = val_acc_metric.result()
        log.info(f'Validation acc:{float(val_acc):.4f}')

        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

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
    val_acc_metric.update(y, val_logits)

    return loss_val

