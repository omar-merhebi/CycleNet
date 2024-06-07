import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from datetime import datetime
from collections.abc import Mapping
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
from typing import Union


CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = datetime.now()
CMAP = plt.cm.viridis

log = logging.getLogger(__name__)
from icecream import ic


class ConfigError(Exception):
    """
    Exception raised for errors in the configuration.
    """
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


def _get_dataset_length(labels_csv: Union[str, Path]) -> int:
    """
    Checks the length of the dataset provided
    Args:
        labels_csv (Union[str, Path]): The CSV file containing the
        dataset labels.

    Returns:
        int: The length of the dataset.
    """
    labels_csv = str(labels_csv)

    labels_csv = pd.read_csv(labels_csv)

    return len(labels_csv)


def generate_save_path(parent_dir: Union[str, Path],
                       model_name: str,
                       file_name: str,
                       extension: str = '.tf',
                       include_time: bool = False) -> str:
    """
    Gets a save path for the results/models.
    Args:
        parent_dir (Union[str, Path]): The directory where all such results are
        stored. ex. save_models/ or results/
        model_name (str): the name of the model (will be a parent directory)
        file_name (str): the name of the file to save (without the extension)
        extension (str, optional): the extension to add to the file name.
        Defaults to '.tf'.
        include_time (bool, optional): whether to include the time in the
        name of the file. Defaults to False.

    Returns:
        str: the output filepath as a string.
    """

    today = TODAY.strftime('%Y-%m-%d')
    now = TODAY.strftime('%H-%M-%S')

    parent_dir = Path(parent_dir)
    extension = extension.replace('.', '')

    save_path = parent_dir / model_name / today

    save_path.mkdir(parents=True, exist_ok=True)

    if include_time:
        file_name = f'{file_name}_{now}.{extension}'

    else:
        file_name = f'{file_name}.{extension}'

    save_path = save_path / file_name

    return save_path


def log_config_error(message: str) -> None:
    """
    Logs a critical configuration error.
    Args:
        message (str): The message to log
    """

    log.critical(message)

    raise ConfigError(message)


def log_cfg(cfg: DictConfig) -> None:
    """
    Logs the Hydra configuration.
    Args:
        cfg (DictConfig): Hydra configuration dict.
    """

    pretty_cfg = json.dumps(cfg, indent=4)

    log.info('----------------------Loaded Config----------------------')
    log.info(f'{pretty_cfg}')


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
            log_config_error(
                "Execution terminated: failed to detect GPU, but "
                "force_gpu is set to True.")
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
        log_config_error(
            'Attempted conversion of image to tensor with non-2D shape.\n'
            f'Invalid image shape: {img_arr.shape}.'
        )

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

    if not cfg.dataset:
        log_config_error('No dataset configuarion found.')

    if not cfg.dataset.data_dir:
        log_config_error('No data directory found.')

    try:
        filters_given = len(cfg.model.filters)

    except TypeError:
        log_config_error(
            'Filters configuration for the model should be a list,'
            f'not {type(cfg.model.filters)}'
        )

    if filters_given != cfg.model.num_conv_layers + 1:
        log_config_error(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{filters_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_conv_layers}"
        )

    try:
        neurons_given = len(cfg.model.dense_neurons)

    except TypeError:
        log_config_error(
            "Dense layer neuron configuration for the model should be a list,"
            f"not {type(cfg.model.dense_neurons)}"
        )

    if neurons_given != cfg.model.num_dense_layers:
        log_config_error(
            "Must provide the number of filters to use for each layer,"
            "i.e., len(filters) == num_conv_layers in model config. Got:\n"
            f"Number of filters:\t{neurons_given}\n"
            f"Number of Conv Layers:\t{cfg.model.num_dense_layers}")


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
