import hydra
import logging
import os
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from src import helpers as h
from src import train as tr
from src import datasets as d
from src.processing import preprocess

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    hydra.initialize(config_path='conf/', version_base='1.1')
    config = hydra.compose('config')
    config_dict = OmegaConf.to_container(config, resolve=True)

    if config.dataset.preprocess:
        preprocess(dataset_name=config.dataset.name,
                   **config_dict['dataset'])
        
        


if __name__ == '__main__':
    main()
