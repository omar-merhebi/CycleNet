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

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None

from icecream import ic


def main():
    with hydra.initialize(config_path='conf/', version_base='1.1'):
        cfg = hydra.compose(config_name='config')

    file_name = (
        f'log/{TODAY.strftime("%Y-%m-%d")}/'
        f'{cfg.wandb.run_name}_{TODAY.strftime("%H-%M-%S")}')

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_dict)
    h.check_config(cfg)
    h.log_cfg(cfg_dict)
    h.test_gpu(cfg.force_gpu)

    cpus, gpus, total_mem = h.get_resource_allocation()
    h.log_env_details(cpus, gpus, total_mem)

    _run(cfg)


def _run(cfg: DictConfig, sweep: bool = False) -> None:
    """
    Detects which run mode to use and runs it
    Args:
        cfg (DictConfig): The full hydra config
        sweep (bool): Whether or not to do a wandb sweep (overrides cfg.mode)
        Defaults to false
    """

    h.init_wandb(cfg)

    mode = cfg.mode.name.lower()

    if mode == 'train':
        train_dataset = d.DATASETS.get(cfg.dataset.name.lower())(
            cfg.dataset, cfg.dataset.train,
        )

    wb.finish()


if __name__ == "__main__":
    main()
