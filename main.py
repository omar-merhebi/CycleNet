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

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose('config')
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if len(sys.argv) > 1 and 'sweep' in sys.argv[1]:
        sweep_id = sys.argv[2]
        wb.init(project=cfg.wandb.project, config=sweep_id)
        sweep_cfg = wb.config
        cfg = update_cfg(cfg, sweep_cfg)

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


def update_cfg(cfg, sweep_cfg):

    for k, v in sweep_cfg.items():
        if isinstance(v, dict):
            cfg[k] = update_cfg[cfg.get(k, {}), v]

        else:
            cfg[k] = v

        return cfg


if __name__ == '__main__':
    main()
