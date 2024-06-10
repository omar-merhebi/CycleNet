import cv2 
import numpy as np
import logging
import pandas as pd
import tensorflow as tf

from pathlib import Path
from typing import Union, Tuple


class WayneCroppedDatawset(tf.keras.utils.Sequence):
    def __init__(self, data_idx, shuffle, balance, batch_size,
                 data_dir, labels, channels, **kwargs):
        self.data_idx = data_idx
        self.shuffle = shuffle
        self.balance = balance
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.channels = channels
        
        labels = pd.read_csv(labels)
        labels = labels.loc[data_idx]
        
        if self.balance:
            if self.balance.lower() == 'middle':
                # Get middle class
                val_counts = labels['pred_phase'].value_counts() 
                
                # For now we just hard code:
                sample_to = val_counts['S']
                
                labels_g1 = resample(
                    labels[labels['G1'] == 1],
                    target=sample_to
                )
                
                labels_s = resample(
                    labels[labels['S'] == 1],
                    target=sample_to
                )
                
                labels_g2 = resample(
                    labels[labels['G2'] == 1],
                    target=sample_to
                )
                
                labels = pd.concat([labels_g1, labels_g2, labels_s])
                
                self.labels = labels
                
        self.indexes = np.arange(labels.shape[0])
                
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def __getitem__(self, idx):
        idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = [self._load_img(i) for i in idx]
        batch_labels = [tf.convert_to_tensor(
            self.labels[['G1', 'S', 'G2']].iloc[i].values) for i in idx]
        
        X = tf.stack(batch_imgs)
        lab = tf.stack(batch_labels)
        
        return X, lab
        
    def _load_img(self, idx):
        filepath = self.labels['filepath'].iloc[idx]
        image = np.load(filepath)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        image = tf.gather(image, self.channels, axis=-1)

        return image
       
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        

def resample(data: Union[pd.DataFrame, pd.Series], target: int):
    """
    Resample the data to the target number of samples.

    Args:
        data (Union[pd.DataFrame, pd.Series]): The data to resample
        target (int): The target number of samples

    Returns:
        pd.DataFrame: The resampled data
    """

    if len(data) > target:
        return data.sample(n=target, replace=False)

    if len(data) < target:
        return data.sample(n=target, replace=True)

    return data
