import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from datetime import date
from icecream import ic
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from PIL import Image
from sklearn.model_selection._split import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from typing import Union
import wandb

from .datasets import WayneRPEDataset
from .model_builder import CustomModel
from .helpers import convert_tensor_to_image

CURRENT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_PATH = CURRENT_PATH.parent
TODAY = date.today().strftime('%Y-%m-%d')

DATASETS = {
    'wayne_rpe': WayneRPEDataset
}

def train(cfg: DictConfig, gpu: bool, num_workers: int = 2) -> None:
    """
    Training loop.
    Args:
        cfg (DictConfig): The configuration object.
    """
    
    wandb.init(project=cfg.wandb.project,
               group=cfg.wandb.group,
               name=cfg.wandb.run_name,
               tags=cfg.wandb.tags,
               config=OmegaConf.to_container(cfg.model, resolve=True))
    
    model_save_path =Path(cfg.model.save_path) / TODAY
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    train_idx, val_idx, test_idx = _create_splits(cfg.dataset.labels, cfg.dataset.splits, cfg.random_seed)
    
    _check_leakage(train_idx, val_idx, test_idx)
    print('No leakage detected between splits.')
    
    dataset = DATASETS.get(cfg.dataset.name)
    if dataset is None:
        raise ValueError(f'Dataset {cfg.dataset.name} not found.')
    
    train_dataset = dataset(cfg, train_idx, augment=cfg.dataset.augment)
    val_dataset = dataset(cfg, val_idx, augment=False)
    
    training_loader = DataLoader(train_dataset, batch_size=cfg.model.train.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.model.train.batch_size, shuffle=False, num_workers=num_workers)
    
    model = CustomModel(cfg.model, input_channels=train_dataset.input_channels, 
                        num_classes=len(train_dataset.unique_phases))
    
    optimizer = _get_optimizer(cfg, model)
    loss_fn = _get_loss_fn(cfg.model.train.loss)
    
    if gpu:
        device = torch.device('cuda')
    
    else:
        device = torch.device('cpu')
        
    print('Model Acrchitecture:\n')
    input_shape = tuple(train_dataset[0][0].shape)
    summary(model, input_size=input_shape, device='cpu')
    
        
    model.to(device)
    wandb.watch(model, log='all', log_freq=100)
        
    best_val_loss = np.inf
    
    for epoch in range(cfg.model.train.epochs):
        # Ensure gradient tracking is on
        model.train(True)
        avg_loss, avg_metric, log_images, cell_ids = train_one_epoch(epoch, model, training_loader, optimizer, 
                                                           loss_fn, cfg.model.train.metric, device)
        
        log_images = [convert_tensor_to_image(img) for img in log_images]
        log_images = log_images[:3] # Only log 3 images per epoch 
        cell_ids = cell_ids[:3]
        
        log_images = [wandb.Image(img, caption=cell_ids[i]) for i, img in enumerate(log_images)]
        
        running_val_loss = 0.0 
        running_val_metric = 0.0
        model.eval()
        
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                val_input, val_labels, val_cell_id, val_imgs = vdata[0].to(device), vdata[1].to(device), vdata[2], vdata[3]
                val_outputs = model(val_input)
                val_loss = loss_fn(val_outputs, val_labels)
                running_val_loss += val_loss.item()
                running_val_metric += calculate_metric(val_outputs, val_labels, cfg.model.train.metric)
                
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_metric = running_val_metric / len(val_loader)
        print(f'LOSS train {avg_loss:.3f} val {avg_val_loss:.3f}')
        print(f'METRIC {cfg.model.train.metric} train {avg_metric:.3f} val {avg_val_metric:.3f}')
        
        wandb.log({"train_loss": avg_loss, "val_loss": avg_val_loss, 
                   f"train_{cfg.model.train.metric}": avg_metric, f"val_{cfg.model.train.metric}": avg_val_metric,
                   "example_images": log_images}, step=epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = model_save_path / f'best_model.pt'
            torch.save(model.state_dict(), model_path)
            wandb.save('best_model.pt')
    
    del train_dataset, val_dataset, training_loader, val_loader
    
    print('Training complete.')

        
def train_one_epoch(epoch: int, model: torch.nn.Module, training_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                    metric: str, device: torch.device) -> float:   
    running_loss = 0.0
    last_loss = 0.0
    running_metric = 0.0
    last_metric = 0.0
    log_images = []
    cell_ids = []

    for i, data in tqdm(enumerate(training_loader), desc=f'Epoch {epoch + 1}',
                        total=len(training_loader)):
        # Load input and labels
        input, labels, train_cell_ids, train_log_imgs = data[0].to(device), data[1].to(device), data[2], data[3]
        
        # Zero parameter gradients
        optimizer.zero_grad()
        
        # Predictions
        outputs = model(input)
        
        # Compute loss and the gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_metric += calculate_metric(outputs, labels, metric)
        
        if i % 100 == 99:
            last_loss = running_loss / 100
            running_loss = 0.0
            last_metric = running_metric / 100
            running_metric = 0.0
            
            log_images.extend([train_log_imgs[i] for i in range(train_log_imgs.shape[0])])
            cell_ids.extend(train_cell_ids)

    return last_loss, last_metric, log_images, cell_ids


def calculate_metric(outputs: torch.Tensor, labels: torch.Tensor, metric: str) -> float:
    """
    Calculate the accuracy of the model.
    Args:
        outputs (torch.Tensor): The model outputs.
        labels (torch.Tensor): The ground truth labels.
    Returns:
        float: The accuracy of the model.
    """
    
    metric = metric.lower()
    
    if metric in ['categorical_accuracy', 'categoricalaccuracy']:
        _, predicted_classes = torch.max(outputs, 1)
        correct = (predicted_classes == labels).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy
                
    
def _get_loss_fn(criterion: str) -> torch.nn.Module:
    """
    Get the loss function based on the string provided.
    Args:
        criterion (str): The loss function to use.

    Returns:
        torch.nn.Module: The loss function.
    """
    
    loss_fns = {
        "crossentropy": torch.nn.CrossEntropyLoss(),
    }
    
    criterion = criterion.lower()
    
    if criterion not in loss_fns:
        raise ValueError(f'Loss function {criterion} not found.')
    
    return loss_fns.get(criterion)
    
    
def _get_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Get the optimizer based on the configuration.
    Args:
        cfg (DictConfig): The model configuration (from cfg.model)
        model (torch.nn.Module): The model to optimize.
    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    
    optimizer = cfg.optimizer.name.lower()
    
    if optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)


def _check_leakage(train_idx: np.ndarray, test_idx: np.ndarray, 
                  val_idx: np.ndarray) -> None:
    """
    Check for leakage between the train, test, and validation splits.
    Args:
        train_idx (np.ndarray): training set indices.
        test_idx (np.ndarray): test set indices.
        val_idx (np.ndarray): validation set indices.
    """
    intersection = set(train_idx) & set(test_idx) | set(train_idx) & set(val_idx) | set(test_idx) & set(val_idx)
    
    assert intersection == set(), f'Leakage detected at index: {intersection}'
    
    
def _create_splits(labels: Union[str, Path], 
                   splits: DictConfig, random_seed: int) -> np.ndarray:
    """
    Create the train, validation, and test splits for the dataset.
    Args:
        labels (Union[str, Path]): Path to the labels CSV file
        splits (DictConfig): The split configuration
        random_seed (int): The random seed to use for reproducibility

    Returns:
        np.ndarray: The indices for the train, validation, and test splits
    """
    
    labels = pd.read_csv(labels)
    
    train_idx, test_idx = train_test_split(np.arange(len(labels)),
                                        train_size=splits.train,
                                        random_state=random_seed)
    
    test_idx, val_idx = train_test_split(test_idx,
                                        train_size=splits.val / (splits.val + splits.test),
                                        random_state=random_seed)
    
    return train_idx, val_idx, test_idx