import torch.nn as nn

from omegaconf import DictConfig
from torchvision import models

def get_model(model_name: str, cfg: DictConfig, input_channels: int=55):
    model_name = model_name.lower()
    
    
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=False)
        