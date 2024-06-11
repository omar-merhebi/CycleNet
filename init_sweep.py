import wandb
import sys

from omegaconf import OmegaConf


def main():
    sweep_config = OmegaConf.load(sys.argv[1])
    sweep_config = OmegaConf.to_container(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])

if __name__ == '__main__':
    main()
