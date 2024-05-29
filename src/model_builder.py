import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

class CustomModel(nn.Module):
    def __init__(self, cfg: DictConfig, input_channels: int, num_classes: int, add_regressor: bool = False):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        
        for layer in cfg.layers:
            layer_type = layer.get('type').lower()
            activation = layer.get('activation', 'relu')
            
            if activation:
                activation = activation.lower()
            
            if layer_type == 'conv2d':
                self.layers.append(nn.Conv2d(in_channels=input_channels,
                                             out_channels=layer.get('out'),
                                             kernel_size=layer.get('kernel_size'),
                                             padding=layer.get('padding', 0),
                                             stride=layer.get('stride', 1)))
                self.layers.append(get_activation(activation))
                input_channels = layer.get('out')
                
            elif layer_type == 'maxpool2d':
                self.layers.append(nn.MaxPool2d(kernel_size=layer.get('pool_size'),
                                                stride=layer.get('stride', None),
                                                padding=layer.get('padding', 0)))
                
            elif layer_type == 'avgpool2d':
                self.layers.append(nn.AvgPool2d(kernel_size=layer.get('pool_size'),
                                                stride=layer.get('stride', None),
                                                padding=layer.get('padding', 0)))
            
            elif layer_type == 'flatten':
                # self.layers.append(nn.AdaptiveAvgPool2d((15, 15)))
                self.layers.append(nn.Flatten())
                input_channels = 12 * 12 * input_channels
                
            elif layer_type == 'dropout':
                self.layers.append(nn.Dropout(p=layer.get('rate', 0.5)))    
            
            elif layer_type in ['linear', 'dense', 'fullyconnected', 'fc']:
                self.layers.append(nn.Linear(in_features=input_channels,
                                            out_features=layer.get('out')))
                self.layers.append(get_activation(activation))
                input_channels = layer.get('out')
                
            self.classifier = nn.Linear(input_channels, num_classes)
            
            if self.add_regressor():
                self.regressor = nn.Linear(input_channels, 1)
                
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        class_output = self.classifier(x)
        
        if self.add_regressor():
            reg_output = self.regressor(x)
            return class_output, reg_output
        
        return class_output
                
                
def get_activation(activation: str) -> nn.Module:
    """
    Get the activation function based on the string provided.
    Args:
        activation (str): The activation function to use.
    """
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leakyrelu': nn.LeakyReLU(inplace=True),
        'softmax': nn.Softmax(dim=1),
    }
    
    return activations.get(activation.lower(), nn.ReLU())
