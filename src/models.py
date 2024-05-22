from omegaconf import DictConfig
from torch import nn

class RotationCNN(nn.Module):
    def __init__(self, train_cfg: DictConfig):
        super(RotationCNN, self).__init__()
        