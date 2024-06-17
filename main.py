import torch
import argparse
import hydra
import os
import tensorflow as tf
import wandb as wb

from datetime import datetime
from pathlib import Path

from src import helpers as h
from src import train as tr


PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
TODAY = datetime.now()
CONF_NAME = None


def main():
    args = parse_args()

    config = Path(args.config).stem

    hydra.initialize(config_path='conf/', version_base='1.1')
    config = hydra.compose(config)

    # Check gpus (also initializes CUDA and CuDNN)
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        print(f'Found {n_gpus} GPUs:')

        for gpu in range(n_gpus):
            print(torch.cuda.get_device_name(gpu))

    if config.force_gpu and not tf.test.is_gpu_available():
        raise RuntimeError('No GPUs found and force gpu is true.')

    if args.mode == 'sweep':
        if args.sweep_config:
            sweep_id = h.init_sweep(config, args.sweep_config)

        else:
            sweep_id = args.sweep_id

        sweep_id = f'{config.wandb.entity}/{config.wandb.project}/{sweep_id}'

        wb.agent(sweep_id,
                 entity=config.wandb.entity,
                 project=config.wandb.project,
                 function=tr.run_sweep,
                 count=1)

    elif args.mode == 'train':
        tr.train(config)


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
