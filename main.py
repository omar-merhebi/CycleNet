import hydra
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from sklearn.model_selection._split import train_test_split

from src import helpers as h
from src import train as tr
from src import model_builder as mb
from src import datasets as d
from src.processing import preprocess

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose('config')
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg.dataset.preprocess:
        preprocess(dataset_name=cfg.dataset.name,
                   **cfg_dict['dataset'])
        
    # Build RPE Dataset
    uni_cfg = cfg.dataset
    train_cfg = cfg.dataset.args.train
    val_cfg  = cfg.dataset.args.val
    test_cfg = cfg.dataset.args.test
    
    splits = [cfg_dict['dataset']['args'][ds]['split'] for ds in 
              cfg_dict['dataset']['args']]
    
    labels = pd.read_csv(cfg.dataset.labels)
    data_len = len(labels)
    
    del labels
    
    train_idx, val_idx, test_idx = _create_data_splits(
        splits, data_len, cfg.random_seed
    )
    
    train_ds = d.WayneCroppedDatawset(
        **uni_cfg, **train_cfg, data_idx=train_idx,
        batch_size=cfg.mode.batch_size)
    
    val_ds = d.WayneCroppedDatawset(
        **uni_cfg, **val_cfg, data_idx=val_idx,
        batch_size=cfg.mode.batch_size)
    
    test_ds = d.WayneCroppedDatawset(
        **uni_cfg, **test_cfg, data_idx=test_idx,
        batch_size=cfg.mode.batch_size
    )
    
    in_shape = train_ds[0][0]._shape
    print(in_shape)
    
    
    

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


if __name__ == '__main__':
    main()
