import argparse
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
    args = parse_args()

    config = Path(args.config).stem

    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose(config)

    if args.mode == 'sweep':
        if args.sweep_config:
            sweep_id = h.init_sweep(cfg, args.sweep_config)

        else:
            sweep_id = args.sweep_id

        sweep_id = f'{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}'
        
        wb.agent(sweep_id,
                 entity=cfg.wandb.entity,
                 project=cfg.wandb.project,
                 function=tr.run_sweep,
                 count=1)


    # if 'WANDB_RUN_ID' in os.environ:
    #     wb.init()
    #     sweep_id = wb.run.sweep_id
    #     cfg = OmegaConf.load(PROJECT_PATH / 'tmp' / f'{sweep_id}.yaml')
    #     sweep_cfg = wb.config
    #     update_config(cfg, sweep_cfg)
    #     tr.run_train(cfg, save_model=False)

    # else:
    #     hydra.initialize(config_path='conf/', version_base='1.1')
    #     cfg = hydra.compose('config')
    #     OmegaConf.set_struct(cfg, False)
    #     cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    #     if cfg.dataset.preprocess:
    #         preprocess(dataset_name=cfg.dataset.name,
    #                    **cfg_dict['dataset'])

    #     if 'inference' in cfg.mode.name:
    #         # // TODO: inference
    #         pass
    #     elif 'train' in cfg.mode.name:
    #         tr.run_train(cfg, save_model=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        choices=['train', 'inference', 'sweep'],
                        required=True,
                        help="Choose one of: train, inference, sweep")
    parser.add_argument('-c', '--config', '--cfg',
                        type=str,
                        required=False,
                        default='config.yaml',
                        help='Path to the config file to use,  '
                        'like: config.yaml. File must be in the conf/ dir. '
                        'Ignored when using sweeps in favor '
                        'of frozen config.')
    parser.add_argument('--sweep-config',
                        type=str,
                        required=False,
                        help="Path to a sweep configuration. "
                        "This file does not need to be in conf/ direcgtory.")
    parser.add_argument('--sweep-id',
                        type=str,
                        required=False,
                        help="Existing wandb sweep ID.")

    args = parser.parse_args()

    if args.mode == 'sweep' and args.sweep_config is None \
            and args.sweep_id is None:
        parser.error('--sweep-config or --sweep-id is required for '
                     'running sweeps.')

    return args


if __name__ == '__main__':
    main()
