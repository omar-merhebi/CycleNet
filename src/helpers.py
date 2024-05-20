import hydra
import os
import torch

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
