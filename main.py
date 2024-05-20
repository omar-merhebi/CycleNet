import hydra
import os
import pandas as pd
import torch

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pprint import pprint as pprint

from src import helpers as hlp

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cpus, gpus, total_mem = hlp.get_resource_allocation()
    
    if not torch.cuda.is_available() or gpus == 0:
        raise RuntimeError('No GPUs available. Aborting...')
    
    print('\nEnvironment Details:')
    print(f'CPUs:\t{cpus}\nGPUs:\t{gpus}\nMemory:\t{total_mem} MB')
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print('\nLoaded Configuration:')
    pprint(cfg)

if __name__ == "__main__":
    main()