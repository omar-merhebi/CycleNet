import matplotlib.pyplot as plt
import os
import tensorflow as tf
import wandb as wb

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union

from icecream import ic


CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = datetime.now()
CMAP = plt.cm.viridis


class ConfigError(Exception):
    """
    Exception raised for errors in the configuration.
    """
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


def init_sweep(config: DictConfig,
               sweep_config: Union[str, Path]):

    sweep_config = Path(sweep_config)

    sweep_config = OmegaConf.load(sweep_config)
    sweep_config = OmegaConf.to_container(sweep_config)

    sweep_id = wb.sweep(sweep_config, project=config.wandb.project)

    _freeze_config(config, fname=sweep_id)

    return sweep_id


def update_config(original, new_params):
    """
    Updates the main config with args passed by wandb sweep config.
    Args:
        original (DictConfig): the original config.
        new_params (DictConfig): the wandb args to replace in the main config.
    """
    for param, new_val, in new_params.items():
        keys = param.split('.')
        current = original
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = new_val


def default_sweep_configs(config):
    """
    Updates some of the config parameters to make sure sweeps run smoothly.
    Args:
        config (DictConfig): the original config.
    """

    to_change = {
        "mode.early_stopping": False,
        "force_gpu": True,
        "dataset.preprocess": False
    }
    
    update_config(config, to_change)


def _freeze_config(config: DictConfig, fname: str):
    """
    Freezes the loaded config. Useful when a run may not start immediately
    when called and we want to be able to change the config in the meantime. 
    For example: in SLURM environments where the config file won't be loaded
    until the job has begun. 
    Args:
        config (DictConfig): _description_
        fname (str): _description_
    """
    save_path = PROJECT_PATH / 'frozen_configs' / f'{fname}.yaml'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, save_path)


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
            raise ConfigError(
                "Execution terminated: failed to detect GPU, but "
                "force_gpu is set to True.")
        else:
            print("Failed to detect GPU. Using CPU instead.")


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
