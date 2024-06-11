import hydra
import os
import wandb
import sys

from omegaconf import OmegaConf
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))


def main():
    sweep_config = OmegaConf.load(sys.argv[1])
    sweep_config = OmegaConf.to_container(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])

    # Freeze the current config:
    hydra.initialize(config_path='conf/', version_base='1.1')
    cfg = hydra.compose('config')

    save_path = PROJECT_PATH / 'tmp' / f'{sweep_id}.yaml'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, save_path)


if __name__ == '__main__':
    main()
