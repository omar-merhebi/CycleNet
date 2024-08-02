import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from . import datasets as d
from . import processing as pr

NOW = datetime.now()
DATE = NOW.strftime('%Y-%m-%d')
TIME = NOW.strftime('%H-%M-%S')


# // TODO: Finish inference logic
def run_inference(config):
    if config.dataset.preprocess:
        pr.preprocess(dataset_name=config.dataset.name,
                      **config.dataset)

    dataset = _load_dataset(config)

    model = Path(config.mode.model)
    model = tf.keras.models.load_model(model)

    results = inference(dataset, model)

    savepath = Path(config.mode.results)
    
    savepath.parent.mkdir(exist_ok=True, parents=True)

    results.to_csv(savepath)


def inference(inference_data, model, cell_id_key='cell_id'):
    colnames = ['id', 'prediction', 'confidence']
    classes = inference_data.classes

    results_df = pd.DataFrame(columns=colnames)

    for s, step in tqdm(enumerate(inference_data),
                        total=len(inference_data)):
        if step[0].shape[0] == 0:
            break

        ids = step[-1][cell_id_key].numpy()

        raw_predictions = model.predict(step[0], verbose=0)
        confidences = np.max(raw_predictions, axis=-1)
        predictions = np.argmax(raw_predictions, axis=-1)
        predictions = np.array([classes[i] for i in predictions])

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
