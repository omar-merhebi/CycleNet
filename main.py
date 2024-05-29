import hydra
import numpy as np
import os
import pandas as pd
import torch
import wandb

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pprint import pprint as pprint

from src import helpers as hlp
from src import train as tr


PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_dict)
    
    hlp.check_config(cfg)
    cfg = hlp.convert_paths(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    cpus, gpus, total_mem = hlp.get_resource_allocation()
    gpus = 1 # until we implement multi-GPU training
    
    if not (torch.cuda.is_available() or gpus == 0) and not cfg.skip_gpu_check:
        raise RuntimeError('No GPUs available. Aborting...')
    
    print('\nEnvironment Details:')
    print(f'CPUs:\t{cpus}\nGPUs:\t{gpus}\nMemory:\t{total_mem} MB')
    
    if gpus == 1:
        print(f'CUDA device: {torch.cuda.get_device_name()}')
        
    else:
        print('CUDA Devices:')
        for i in range(gpus):
            print(f'\tDevice {i}: {torch.cuda.get_device_name(i)}')
            
    print('\nLoaded Configuration:')
    pprint(cfg_dict)
    
    model_cfg = OmegaConf.to_container(cfg.model)
    num_workers = (cpus - 2) // 2 # leave 2 CPUs for the OS/training
    
    if cfg.mode.lower() == 'train':
        tr.train(cfg, gpu=torch.cuda.is_available(), num_workers=num_workers)

if __name__ == "__main__":
    main()