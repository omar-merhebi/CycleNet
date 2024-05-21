import hydra
import numpy as np
import os
import pandas as pd
import torch

from datetime import date
from collections.abc import Mapping
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Union

from .datasets import *

CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = date.today().strftime('%Y-%m-%d')

def train(cfg):
    model_save_path = f'{Path(cfg.model.save_path) / TODAY}'
    
    train_idx, val_idx, test_idx = _create_splits(cfg.dataset.labels, cfg.mode.splits, cfg.random_seed)
    
    train_dataset = WayneRPEDataset(cfg, train_idx)
    ic(train_dataset.__getitem__(0))

def _create_splits(labels: Union[str, Path], 
                   splits: DictConfig, random_seed: int) -> np.ndarray:
    """
    Create the train, validation, and test splits for the dataset.
    Args:
        labels (Union[str, Path]): Path to the labels CSV file
        splits (DictConfig): The split configuration
        random_seed (int): The random seed to use for reproducibility

    Returns:
        np.ndarray: The indices for the train, validation, and test splits
    """
    
    labels = pd.read_csv(labels)
    
    train_idx, test_idx = train_test_split(np.arange(len(labels)),
                                        train_size=splits.train,
                                        random_state=random_seed)
    
    test_idx, val_idx = train_test_split(test_idx,
                                        train_size=splits.val / (splits.val + splits.test),
                                        random_state=random_seed)
    
    return train_idx, val_idx, test_idx
    


def get_resource_allocation():
    """
    Get the resource allocation for the current environment.
    Returns:
        cpus (int): The number of CPUs available
        gpus (int): The number of GPUs available
        total_mem (int): The total memory available in MB
    """
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


def check_config(cfg: DictConfig):
    """
    Check the configuration object for any missing or invalid values.
    
    Args:
        cfg (DictConfig): The configuration object
    """
    
    split_names = ['train', 'val', 'test']
    
    if not cfg.dataset:
        raise ValueError('No dataset configuration found.')
    
    if not cfg.dataset.data_dir:
        raise ValueError('No data directory found.')
    
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
    
    
def convert_paths(cfg: DictConfig):
    if isinstance(cfg, Mapping):
        for key, value in cfg.items():
            if isinstance(value, str):
                cfg[key] = _to_posix(value)
                
            elif isinstance(value, list):
                cfg[key] = [_to_posix(v) if isinstance(v, str) and ('/' in v or '\\' in v) else v for v in value]
                
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
    """_summary_

    Args:
        path (Union[str, Path]): _description_
        project_path (Path, optional): _description_. Defaults to PROJECT_PATH.
    """
    
    if '/' in path or '\\' in path:
        new_path = Path(path)
        
        if not new_path.is_absolute() and '~' not in path:
            new_path = project_path / new_path
            
        return new_path
    
    return path