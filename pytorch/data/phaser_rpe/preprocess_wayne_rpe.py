"""
Script to preporcess the PHASER RPE dataset. The script does the following:
1. Removes cells with missing labels from labels.csv
2. Converts cell id in labels.csv to match image file names. 
"""

import argparse
import hydra
import pandas as pd

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

CONFIG_PATH = "../../conf/dataset/"

def main():
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="wayne_rpe")
        
    # Load labels
    labels = pd.read_csv(Path(cfg.labels))
    print(f"Original labels shape: {labels.shape}")
    labels = labels.dropna()
    print(f"Labels shape after removing missing values: {labels.shape}")
    
    # Convert cell id to match image file names
    labels["cell_id"] = labels["cell_id"].apply(lambda x: f"cell_{x:04d}")
    
    # Save labels
    labels.to_csv(Path(cfg.labels), index=False)

if __name__ == "__main__":
    main()