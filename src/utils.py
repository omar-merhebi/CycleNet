import os
import numpy as np
import pandas as pd
import torch

from datetime import date
from icecream import ic
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection._split import train_test_split
from torch.utils.data import DataLoader
from typing import Union

from .datasets import WayneRPEDataset

CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = date.today().strftime('%Y-%m-%d')

DATASETS = {
    'wayne_rpe': WayneRPEDataset
}

def train(cfg: DictConfig) -> None:
    """
    Training loop.
    Args:
        cfg (DictConfig): The configuration object.
    """
    model_save_path = f'{Path(cfg.model.save_path) / TODAY}'
    
    train_idx, val_idx, test_idx = create_splits(cfg.dataset.labels, cfg.mode.splits, cfg.random_seed)
    
    check_leakage(train_idx, val_idx, test_idx)
    print('No leakage detected between splits.')
    
    dataset = DATASETS.get(cfg.dataset.name)
    if dataset is None:
        raise ValueError(f'Dataset {cfg.dataset.name} not found.')
    
    train_dataset = dataset(cfg, train_idx, augment=cfg.dataset.augment)
    val_dataset = dataset(cfg, val_idx, augment=cfg.dataset.augment)
    test_dataset = dataset(cfg, test_idx, augment=False)
    
    training_loader = DataLoader(train_dataset, batch_size=cfg.mode.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.mode.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.mode.batch_size, shuffle=False)
    
    ic(train_dataset.unique_phases)
    

def check_leakage(train_idx: np.ndarray, test_idx: np.ndarray, 
                  val_idx: np.ndarray) -> None:
    """
    Check for leakage between the train, test, and validation splits.
    Args:
        train_idx (np.ndarray): training set indices.
        test_idx (np.ndarray): test set indices.
        val_idx (np.ndarray): validation set indices.
    """
    intersection = set(train_idx) & set(test_idx) | set(train_idx) & set(val_idx) | set(test_idx) & set(val_idx)
    
    assert intersection == set(), f'Leakage detected at index: {intersection}'
    
    
def create_splits(labels: Union[str, Path], 
                   splits: DictConfig, random_seed: int) -> np.ndarray:
    """
    Create the train, validation, and test splits for the dataset.
    Args:
        labels (Union[str, Path]): Path to the labels CSV file
        splits (DictConfig): The split configuration
        random_seed (int): The random seed to use for reproducibility

    Returns:
        np.ndarray: The indices for the train, validation, and test splits
    """
    
    labels = pd.read_csv(labels)
    
    train_idx, test_idx = train_test_split(np.arange(len(labels)),
                                        train_size=splits.train,
                                        random_state=random_seed)
    
    test_idx, val_idx = train_test_split(test_idx,
                                        train_size=splits.val / (splits.val + splits.test),
                                        random_state=random_seed)
    
    return train_idx, val_idx, test_idx