import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from pathlib import Path

from . import datasets as d

NOW = datetime.now()
DATE = NOW.strftime('%Y-%m-%d')
TIME = NOW.strftime('%H-%M-%S')


# // TODO: Finish inference logic
def run_inference(config):
    dataset = _load_dataset(config)

    model = Path(config.mode.model)
    model = tf.keras.models.load_model(model)

    results = inference(dataset, model)

    savepath = Path(config.mode.results)

    results.to_csv(savepath)


def inference(inference_data, model, cell_id_key='cell_id'):
    colnames = ['id', 'prediction', 'confidence']
    labels = inference_data.labels

    results_df = pd.DataFrame(columns=colnames)

    for s, step in enumerate(inference_data):
        ids = step[-1][cell_id_key].numpy()

        raw_predictions = model.predict(step[0], verbose=0)
        confidences = np.max(raw_predictions, axis=-1)
        predictions = np.argmax(raw_predictions, axis=-1)
        predictions = np.array([labels[i] for i in predictions])

        to_stack = [ids, predictions, confidences]

        stacked = np.vstack(to_stack)
        temp_df = pd.DataFrame(stacked.T, columns=colnames)

        results_df = pd.concat([results_df, temp_df], ignore_index=True)

    return results_df


def _load_dataset(config):
    dataset = d.WayneCroppedDataset(
        shuffle=False,
        balance=False,
        batch_size=1,
        data_dir=config.dataset.data_dir,
        labels=config.dataset.data_csv,
        channels=config.dataset.channels
    )

    return dataset