import cv2
import numpy as np
import os
import pandas as pd
import tifffile as tiff
import torch

from collections import Counter
from icecream import ic
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Union

class WayneRPEDataset(Dataset):
    def __init__(self, cfg: DictConfig, data_idx: np.ndarray):
        self.cfg = cfg
        self.data_idx = data_idx
        self.prediction = cfg.model.predict.lower() if cfg.model.predict else 'both'
        
        self.channel_annotations = pd.read_csv(cfg.dataset.channel_annotations)
        self.channel_annotations['feature'] = self.channel_annotations['feature'].apply(lambda x: x.lower().strip())
        
        self.use_channels = cfg.dataset.use_channels
        self.use_channels = [channel.lower().strip() for channel in self.use_channels]
        
        self.labels = pd.read_csv(cfg.dataset.labels)
        self.labels['phase_index'], unique_phases = pd.factorize(self.labels['pred_phase'])
        self.labels = self.labels.iloc[self.data_idx].reset_index(drop=True)
        
        if self.cfg.dataset.balancing:
            balancing = self.cfg.dataset.balancing.lower()
            class_counts = Counter(self.labels['pred_phase'])
            
            if 'over' in balancing or 'up' in balancing:
                target = class_counts['G1']
                
            elif 'under' in balancing or 'down' in balancing:
                target = class_counts['G2']
                
            elif 'middle' in balancing or 'even' in balancing:
                target = class_counts['S']
                
            labels_g1 = _resample(self.labels[self.labels['pred_phase'] == 'G1'], target)
            labels_s = _resample(self.labels[self.labels['pred_phase'] == 'S'], target)
            labels_g2 = _resample(self.labels[self.labels['pred_phase'] == 'G2'], target)
            labels_m = self.labels[self.labels['pred_phase'] == 'M'] # Don't resample M (too few samples)

            self.labels = pd.concat([labels_g1, labels_s, labels_g2, labels_m])
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        filepath = self.cfg.dataset.data_dir / f'{self.labels.iloc[idx]["cell_id"]}.tif'
        tiff_stack = tiff.imread(filepath)
        tiff_tensor = torch.tensor(tiff_stack)
        
        # Take out non-nucleous or -ring masks
        tiff_tensor = torch.cat((tiff_tensor[:55], tiff_tensor[57:]), dim=0)
        tiff_tensor = _normalize_image(tiff_tensor)
        
        if self.cfg.dataset.use_masks:
            use_masks = self.cfg.dataset.use_masks
            nuc_mask = tiff_tensor[55].bool()
            ring_mask = tiff_tensor[56].bool()
            combined_mask = torch.logical_or(nuc_mask, ring_mask)
            
            masks = [nuc_mask, ring_mask, combined_mask]
            
            # Find the mask closest to the center of the image
            masks_centered = [_find_center_mask(mask) for mask in masks]
            
            if 'nuc' in use_masks:
                tiff_tensor = tiff_tensor * masks_centered[0]
                
            elif 'ring' in use_masks:
                tiff_tensor = tiff_tensor * masks_centered[1]
                
            elif 'both' in use_masks or 'combined' in use_masks:
                tiff_tensor = tiff_tensor * masks_centered[2]
                                
        if self.cfg.dataset.use_channels:
            channel_idx = list(set(self.channel_annotations[self.channel_annotations['feature'].isin(self.use_channels)]['frame'].tolist()))
            ic(channel_idx)
            tiff_tensor_filtered = tiff_tensor[channel_idx]
            
            # Double-check proper slicing (probably not needed in production code)
            for c, channel in enumerate(channel_idx):
                assert torch.equal(tiff_tensor_filtered[c], tiff_tensor[channel])
                
            tiff_tensor = tiff_tensor_filtered
        
        else:
            tiff_tensor = tiff_tensor[:55]
        
        if self.prediction == 'phase':
            return tiff_tensor, self.labels.iloc[idx]['phase_index']
        
        if self.prediction == 'age':
            return tiff_tensor, self.labels.iloc[idx]['age']
        
        return tiff_tensor, self.labels.iloc[idx]['phase_index'], self.labels.iloc[idx]['age']
        
        
def _resample(data: Union[pd.DataFrame, pd.Series], target: int):
    """
    Resample the data to the target number of samples.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): The data to resample
        target (int): The target number of samples
    
    Returns:
        pd.DataFrame: The resampled data
    """
    
    if len(data) > target:
        return data.sample(n=target, replace=False)
    
    if len(data) < target:
        return data.sample(n=target, replace=True)
    
    return data


def _find_center_mask(mask: Union[torch.Tensor, np.ndarray]):
    mask_unit8 = mask.numpy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_unit8)
    img_center = np.array(mask.shape) // 2
    
    distances = np.linalg.norm(centroids - img_center, axis=1)
    closest_label = np.argmin(distances[1:]) + 1
    center_mask = (labels == closest_label).astype(np.uint8)
    
    return torch.tensor(center_mask)


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    return image / 255