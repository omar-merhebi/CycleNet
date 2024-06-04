import matplotlib.pyplot as plt
import json
import logging
import numpy as np
import os
import tensorflow as tf

from datetime import date
from collections.abc import Mapping
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
from typing import Union

CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = date.today().strftime('%Y-%m-%d')
CMAP = plt.cm.viridis

log = logging.getLogger(__name__)


def log_cfg(cfg: DictConfig) -> None:
    """
    Logs the Hydra configuration.
    Args:
        cfg (DictConfig): Hydra configuration dict.
    """

    pretty_cfg = json.dumps(cfg, indent=4)

    log.info(f'Loaded Config:\n{pretty_cfg}')


def log_env_details(cpus: int, gpus: int, total_mem: int) -> None:
    """
    Logs environment details at level INFO.
    Args:
        cpus (int): Number of CPUs
        gpus (int): Number of GPUs
        total_mem (int): Amount of Memory in MB
    """
    log.info(f"CPUs:\t{cpus}")
    log.info(f"GPUs:\t{gpus}")
    log.info(f"Available Memory:\t{total_mem} MB")


def test_gpu(force_gpu: bool = False) -> None:
    """
    Tests whether the gpu is available and logs. Stops execution if
    force_gpu = True but one isn't available.
    Args:
        force_gpu (bool, optional): Whether to force execution halt
        if gpu not available. Defaults to False.
    """

    if not tf.test.is_gpu_available:
        if force_gpu:
            log.critical("Execution terminated: failed to detect GPU, but "
                         "force_gpu is set to True.")
            raise RuntimeError

        else:
            log.warning("Failed to detect GPU. Using CPU instead.")


def convert_tensor_to_image(img_tensor: tf.Tensor) -> Image:
    """
    Convert a TensorFlow tensor to a Image.
    Args:
        img_tensor (tf.Tensor): The image tensor

    Returns:
        Image: Output image
    """

    if img_tensor.dtype != tf.float32:
        img_tensor = tf.cast(img_tensor, tf.float32)

    img_arr = img_tensor.numpy()
    img_arr = img_arr.squeeze()

    if len(img_arr.shape) != 2:
        log.critical(
            'Attempted conversion of image to tensor with non-2D shape.\n'
            f'Invalid image shape: {img_arr.shape}.')

        raise ValueError(
            'Attempted conversion of image to tensor with non-2D shape.\n'
            f'Invalid image shape: {img_arr.shape}.')

    img_min = img_arr.min()
    img_max = img_arr.max()

    if img_min == img_max:
        return Image.fromarray(np.zeros(img_arr.shape))

    normalized_img = CMAP((img_arr - img_min) / (img_max - img_min))
    image = (normalized_img[:, :, :3]*255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def get_resource_allocation():
    """
    Get the resource allocation for the current environment.
    Returns:
        cpus (int): The number of CPUs available
        gpus (int): The number of GPUs available
        total_mem (int): The total memory available in MB
    """
    is_slurm = _determine_slurm()
    gpu_devices = tf.config.list_physical_devices('GPU')
    gpus = len(gpu_devices)

    if is_slurm:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
        mem_per_cpu = os.getenv('SLURM_MEM_PER_CPU')

        if mem_per_cpu:
            mem_per_cpu = int(mem_per_cpu.replace('M', ''))
            total_mem = cpus * mem_per_cpu

        else:
            total_mem = round(
                int(os.environ.get('SLURM_MEM_PER_NODE').replace('M', '')))

    else:
        cpus = os.cpu_count()
        total_mem = (round(os.sysconf('SC_PAGE_SIZE') *
                           os.sysconf('SC_PHYS_PAGES') /
                           (1024. ** 3)))

    return cpus, gpus, total_mem


def _determine_slurm():
    """
    Determines if the current environment is a SLURM environment.
    """
    return os.environ.get('SLURM_JOB_ID') is not None


def check_config(cfg: DictConfig):
    """
    Check the configuration object for any missing or invalid values.

    Args:
        cfg (DictConfig): The configuration object
    """

    split_names = ['train', 'val', 'test']

    if not cfg.dataset:
        log.critical('No dataset configuarion found.')
        raise ValueError('No dataset configuration found.')

    if not cfg.dataset.data_dir:
        log.critical('No data directory found.')
        raise ValueError('No data directory found.')

    if not cfg.dataset.splits:
        log.critical('No splits configuation found.')
        raise ValueError('No splits configuration found.')

    if cfg.dataset.mask:
        if cfg.dataset.mask not in ['nuc', 'cell']:
            log.critical(f'Invalid mask name: {cfg.dataset.mask}')
            raise ValueError(f'Invalid mask name: {cfg.dataset.mask}')

    total_split = 0
    for (k, v) in cfg.dataset.splits.items():
        if not v:
            log.critical('No value found for split {k}.')
            raise ValueError(f'No value found for split {k}.')

        if k not in split_names:
            log.critical(f'Invalid split name: {k}.'
                         f'Must be one of: {split_names}.')

            raise ValueError(f'Invalid split name: {k}.'
                             f'Must be one of: {split_names}.')

        total_split += v

    if total_split != 1:
        log.critical('Split values must sum to 1.')
        raise ValueError('Split values must sum to 1.')

    try:
        filters_given = len(cfg.model.filters)

    except TypeError:
        log.critical(
            'Filters configuration for the model should be a list,'
            f'not {type(cfg.model.filters)}')

        raise ValueError(
            'filters configuration for the model should be a list,'
            f'not {type(cfg.model.filters)}')
        
    if filters_given != cfg.model.num_conv_layers:
        log.critical(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{filters_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_conv_layers}")
        
        raise ValueError(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{filters_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_conv_layers}"
        )
        
    try:
        neurons_given = len(cfg.model.dense_neurons)
        
    except TypeError:
        log.critical(
            "Dense layer neuron configuration for the model should be a list,"
            f"not {type(cfg.model.dense_neurons)}"
        )
        
        raise ValueError(
            "Dense layer neuron configuration for the model should be a list,"
            f"not {type(cfg.model.dense_neurons)}"
        )
        
    if neurons_given != cfg.model.num_dense_layers:
        log.critical(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{neurons_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_dense_layers}")
        
        raise ValueError(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{filters_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_conv_layers}"
        )


def convert_paths(cfg: DictConfig) -> DictConfig:
    """
    Convert all paths in the configuration object to Posix paths.
    Args:
        cfg (DictConfig): The configuration object

    Returns:
        DictConfig: The configuration object with all
        paths converted to Posix paths.
    """

    if isinstance(cfg, Mapping):
        for key, value in cfg.items():
            if isinstance(value, str):
                cfg[key] = _to_posix(value)

            elif isinstance(value, list):
                cfg[key] = [_to_posix(v) if isinstance(v, str) and
                            ('/' in v or '\\' in v)
                            else v for v in value]

            else:
                convert_paths(value)

    elif isinstance(cfg, list):
        for i, value in enumerate(cfg):
            if isinstance(value, str):
                if '/' in value or '\\' in value:
                    cfg[i] = _to_posix(value)

            elif isinstance(value, Mapping):
                cfg[i] = _to_posix(value)

    return cfg


def _to_posix(path: Union[str, Path], project_path: Path = PROJECT_PATH):
    """
    Convert a path to a Posix path.
    Args:
        path (Union[str, Path]): The path to convert
        project_path (Path, optional): project directory.
        Defaults to PROJECT_PATH.

    Returns:
        Path: a path object
    """

    if '/' in path or '\\' in path:
        new_path = Path(path)

        if not new_path.is_absolute() and '~' not in path:
            new_path = project_path / new_path

        return new_path

    return path
