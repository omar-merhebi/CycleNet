import pandas as pd
import tensorflow as tf

from pathlib import Path

def inference(config, inference_data, model, test_acc_metric, sample_id_ind):
    model_path = Path(config.mode.model)
    
    model = tf.keras.models.load_model(model_path)
    
    


def inference_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_val
