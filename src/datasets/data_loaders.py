import hydra
import numpy as np
import os
import pandas
import pickle
import re

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from typing import Union

class RPEDataset(Dataset):
    """ RPE Dataset class """
    
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg (DictConfig): Hydra configuration object
        """
        
        self.cfg = cfg
        