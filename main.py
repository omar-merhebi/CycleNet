import torch
import argparse
import hydra
import os
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

    cpus, gpus, total_mem = h.get_resource_allocation()

    print('\n\n---------------------Environment Details---------------------')
    print(f'CPU Count:\t{cpus}\nGPU Count:\t{gpus}'
          f'\nTotal Memeory:\t{total_mem} MB')

    if gpus == 0 and config.force_gpu:
        raise RuntimeError('No GPU found and force_gpu is set to True.')

    print('\nGPU Details:')

    for gpu in range(gpus):
        device_name = torch.cuda.get_device_name(gpu)
        print(f'    -  {device_name}')

    print(f'\nCUDA Version:\t{torch.version.cuda}')
    print(f'CuDNN Version:\t{torch.backends.cudnn.version()}\n\n')

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
