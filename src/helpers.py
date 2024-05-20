import hydra
import os
import pandas as pd
import torch

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union

def get_resource_allocation():
    is_slurm = _determine_slurm()
    
    if is_slurm:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
        gpus = 0 if not os.environ.get('SLURM_JOB_GPUS') else int(os.environ.get('SLURM_JOB_GPUS'))
        
        mem_per_cpu = os.getenv('SLURM_MEM_PER_CPU')
        if mem_per_cpu:
            mem_per_cpu = int(mem_per_cpu.replace('M', ''))
            total_mem = cpus * mem_per_cpu
        else:
            total_mem = round(int(os.environ.get('SLURM_MEM_PER_NODE').replace('M', '')))
            
    else:
        cpus = os.cpu_count()
        gpus = 0 if not torch.cuda.is_available() else torch.cuda.device_count()
        total_mem = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3))
            
    return cpus, gpus, total_mem
    

def _determine_slurm():
    """
    Determines if the current environment is a SLURM environment.
    """
    return os.environ.get('SLURM_JOB_ID') is not None


def _resample(data: Union[pd.DataFrame, pd.Series], target: int):
    """
    Resample the data to the target number of samples.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): The data to resample
        target (int): The target number of samples
    
    Returns:
        pd.DataFrame: The resampled data
    """
    
    if len(data) > target:
        return data.sample(n=target, replace=False)
    
    if len(data) < target:
        return data.sample(n=target, replace=True)
    
    return data


def check_config(cfg: DictConfig):
    """
    Check the configuration object for any missing or invalid values.
    
    Args:
        cfg (DictConfig): The configuration object
    """
    
    split_names = ['train', 'val', 'test']
    
    if not cfg.dataset:
        raise ValueError('No dataset configuration found.')
    
    if not cfg.mode:
        raise ValueError('No mode configuration found.')
    
    if not cfg.mode.splits:
        raise ValueError('No splits configuration found.')
    
    total_split = 0
    for (k,v) in cfg.mode.splits.items():
        if not v:
            raise ValueError(f'No value found for split {k}.')
        
        if k not in split_names:
            raise ValueError(f'Invalid split name: {k}. Must be one of: {split_names}.')
        
        total_split += v
        
    if total_split != 1:
        raise ValueError('Split values must sum to 1.')