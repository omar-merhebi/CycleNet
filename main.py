import hydra
import os
import pandas as pd

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pprint import pprint as pprint

from src import helpers as hlp

PROJECT_PATH = Path(os.paath.dirname(os.path.realpath(__file__)))

@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    pass

if __name__ == "__main__":
    main()