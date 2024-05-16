import hydra
import os

from pathlib import Path

def convert_cfg_paths_to_posix(cfg, project_dir):
    for item in cfg.dataset:
        try:
            if cfg.dataset[item].startswith('/') or cfg.dataset[item].startswith('\\'):
                cfg.dataset[item] = project_dir / cfg.dataset[item]
                
            elif '/' in cfg.dataset[item] or '\\' in cfg.dataset[item]:
                cfg.dataset[item] = Path(cfg.dataset[item])
                
        except TypeError:
            pass