import cv2
import numpy as np
import os
import pandas as pd
import tifffile as tiff
import torch

from collections import Counter
from icecream import ic
from omegaconf import DictConfig
from PIL import Image
from random import randint
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union

class WayneRPEDataset(Dataset):
    def __init__(self, cfg: DictConfig, data_idx: np.ndarray, augment: bool = False):
        self.cfg = cfg
        self.data_idx = data_idx
        self.augment = augment
        
        self.channel_annotations = pd.read_csv(cfg.dataset.channel_annotations)
        self.channel_annotations['feature'] = self.channel_annotations['feature'].apply(lambda x: x.lower().strip())
        
        self.use_channels = [channel.lower().strip() for channel in self.cfg.dataset.use_channels] if self.cfg.dataset.use_channels else None
        self.input_channels = len(self.use_channels) if self.use_channels else 55
        
        self.labels = pd.read_csv(cfg.dataset.labels)
        self.labels['phase_index'], self.unique_phases = pd.factorize(self.labels['pred_phase'])
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

            self.labels = pd.concat([labels_g1, labels_s, labels_g2])
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        filepath = self.cfg.dataset.data_dir / f'{self.labels.iloc[idx]["cell_id"]}.tif'
        image = tiff.imread(filepath)
        image = torch.tensor(image, dtype=torch.float32)
        
        image = _normalize_image(image)
    
        # We extract and center masks even if we don't use them in the model
        nuc = image[57].bool()
        ring = image[58].bool()
        combined = torch.logical_or(nuc, ring)

        orig_masks = [nuc, ring, combined]

        centered_masks = [_find_center_mask(mask) for mask in orig_masks]
        all_masks = orig_masks + centered_masks
        masks_to_add = tuple(mask.unsqueeze(0) for mask in all_masks[2:])
        image = torch.cat((image, *masks_to_add), dim=0)
        
        if self.augment:
            image = self.augmentations(image)
            
        original_image = image.clone()
        
        if self.cfg.dataset.use_masks:
            use_masks = self.cfg.dataset.use_masks
            
            if 'nuc' in use_masks:
                image = image[60] * image
                
            if 'ring' in use_masks:
                image = image[61] * image
                
            else:
                image = image[62] * image
                
        if self.cfg.dataset.fill.enabled:
            if self.cfg.dataset.fill.fill_cell:
                exclude = ~(original_image[59] != original_image[62])
                image = image * exclude
            
            mask = ~original_image[59].bool() # invert the mask
            sample_from = original_image * mask
            sample_vals = [sample_from[z][sample_from[z] != 0].flatten() for z in range(sample_from.shape[0])]
            zero_indices = (image == 0)
            
            for z in range(image.shape[0] - 6):
                zero_idx = torch.nonzero(zero_indices[z], as_tuple=True)
                random_vals = sample_vals[z][torch.randint(0, sample_vals[z].size(0), (zero_idx[0].numel(), ))]
                
                image[z][zero_idx] = random_vals
                
        if self.cfg.dataset.log_image: 
            channel_idx = list(set(self.channel_annotations[self.channel_annotations['feature'].isin(self.use_channels)]['frame'].tolist()))
            assert len(channel_idx) == 1, f'Can only log one image to wandb. Found {len(channel_idx)} images to log.'
            
            log_image = image[channel_idx]
            log_image = log_image.permute(1, 2, 0)
            
        else:
            log_image = None
                     
        if self.cfg.dataset.use_channels:
            channel_idx = list(set(self.channel_annotations[self.channel_annotations['feature'].isin(self.use_channels)]['frame'].tolist()))
            
            assert len(channel_idx) == self.input_channels, f'The names of one or more channels provided in the configuration file do not match those found in the channel annotations file: {self.cfg.dataset.channel_annotations}'
            
            image_filtered = image[channel_idx]
            
            # Double-check proper slicing (probably not needed in production code)
            for c, channel in enumerate(channel_idx):
                assert torch.equal(image_filtered[c], image[channel])
                
            image = image_filtered
            
            
        else:
            image = image[:55]
            
        
        return image, self.labels.iloc[idx]['phase_index'], self.labels.iloc[idx]['cell_id'], log_image
    
    
    def augmentations(self, image: torch.Tensor) -> torch.Tensor:
        """
        Method to augment the image by random rotation and translation. 
        Args:
            image (torch.Tensor): The image to augment.
        """
        assert type(image) == torch.Tensor, f'The input must be a torch.Tensor, not {type(image)}.'
        
        # Random rotation
        angle = np.random.randint(0, 360)
        image = transforms.functional.rotate(image, angle)
        
        xshift_min, xshift_max, yshift_min, yshift_max = _get_min_max_axis(image[62])

        xshift = np.random.randint(xshift_min, xshift_max)
        yshift = np.random.randint(yshift_min, yshift_max)

        image = torch.roll(image, shifts=(xshift, yshift), dims=(1, 2))
        
        return image

        
def _get_min_max_axis(arr: torch.Tensor) -> tuple:
    """
    Finds the extreme points of a binary mask along the x and y axes.
    
    Args:
        arr (torch.Tensor): the binary mask as a torch.Tensor

    Returns:
        tuple: tuple of the minimum and maximum values along the x and y axes (min_x, max_x, min_y, max_y)
    """
        
    assert type(arr) == torch.Tensor, 'The input must be a torch.Tensor.'
    
    arr_rot = torch.rot90(arr, k=1, dims=[1,0])
    
    posx = [i for i, row in enumerate(arr_rot) if torch.any(row > 0)]
    posy = [i for i, col in enumerate(arr) if torch.any(col > 0)]
    
    return -1 * posx[0], arr.shape[0] - posx[-1], -1 * posy[0], arr.shape[1] - posy[-1]
        
        
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


def _find_center_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Finds the center-most mask in a binary mask.
    Args:
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: returns a new tensor with only the center-most mask.
    """
    assert type(mask) == torch.Tensor, f'The input must be a torch.Tensor, not {type(mask)}.'
    
    mask_unit8 = mask.numpy().astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_unit8)
    img_center = np.array(mask.shape) // 2
    
    distances = np.linalg.norm(centroids - img_center, axis=1)
    closest_label = np.argmin(distances[1:]) + 1
    center_mask = (labels == closest_label).astype(np.uint8)
    
    return torch.tensor(center_mask)


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.float32:
        image = image.to(torch.int32)
        
    return image.float() / 65535.0