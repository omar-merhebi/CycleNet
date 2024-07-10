import tensorflow as tf
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

    devices = tf.config.list_physical_devices('GPU')
    if devices:
        print('\nGPU Details:')
        for d, device in enumerate(devices):
            details = tf.config.experimental.get_device_details(devices[d])
            name = details.get('device_name', 'Unknown')
            print(f'    - {name}')

            # set memory growth
            tf.config.experimental.set_memory_growth(device, True)

    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
    print(f'\nCUDA Version:\t{cuda_version}')
    print(f'cuDNN Version:\t{cudnn_version}\n\n')
    print('Checking ability to use matrix multiplication...')
    result = simple_op()
    print(f'Result of matrix multiplication:\n{result}\n\n')

    if args.mode == 'sweep':
        if args.sweep_config:
            sweep_id = h.init_sweep(config, args.sweep_config)

        else:
            sweep_id = args.sweep_id

        sweep_id = f'{config.wandb.entity}/{config.wandb.project}/{sweep_id}'

        wb.agent(sweep_id,
                 entity=config.wandb.entity,
                 project=config.wandb.project,
                 function=tr.run_sweep)

    elif args.mode == 'train':
        tr.setup_training(config)

    wb.finish()

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


@tf.function
def simple_op():
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        return tf.matmul(a, b)


if __name__ == '__main__':
    main()
