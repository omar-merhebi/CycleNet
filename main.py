import hydra
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from src import helpers as h
from src import train as tr
from src import model_builder as mb
from src import datasets as d
from src.processing import preprocess

import pprint

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose('config')
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if 'WANDB_RUN_ID' in os.environ:
        wb.init()
        sweep_cfg = wb.config
        update_config(cfg, sweep_cfg)
        tr.run_train(cfg, save_model=False)

    else:
        if cfg.dataset.preprocess:
            preprocess(dataset_name=cfg.dataset.name,
                       **cfg_dict['dataset'])

        if 'inference' in cfg.mode.name:
            # // TODO: inference
            pass
        elif 'train' in cfg.mode.name:
            tr.run_train(cfg, save_model=True)


def update_config(original, new_params):
    for param, new_val, in new_params.items():
        keys = param.split('.')
        current = original
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = new_val


if __name__ == '__main__':
    main()
