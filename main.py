import hydra
import os
import pandas as pd

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pprint import pprint as pprint

from src import helpers as hlp

PROJECT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print('\nLoaded Configuration:\n')
    pprint(cfg)

if __name__ == "__main__":
    main()