import cv2
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
import tifffile as tiff

from collections import Counter
from omegaconf import DictConfig
from scipy import ndimage
from typing import Union, Tuple

from . import helpers as h

log = logging.getLogger(__name__)


def get_min_max_axis(img: tf.Tensor) -> Tuple[int, int, int, int]:
    """
    Finds the extreme points of a binary mask along the x and y axes.
    Args:
        img (tf.Tensor): the binary mask as a torch.Tensor

    Returns:
        tuple: tuple of the minimum and maximum values along the x and y axes
               (ymin, ymax, xmin, xmax)
    """

    assert isinstance(img, tf.Tensor), (
        f"Input must be a tf.Tensor, not {type(img)}")

    nonzero = tf.where(tf.not_equal(img, 0))

    xidx = tf.unique(nonzero[:, 1]).y
    yidx = tf.unique(nonzero[:, 0]).y

    xmin = -1 * tf.reduce_min(xidx).numpy()
    ymin = -1 * tf.reduce_min(yidx).numpy()

    xmax = img.shape[0] - tf.reduce_max(xidx).numpy()
    ymax = img.shape[1] - tf.reduce_max(yidx).numpy()

    return ymin, ymax, xmin, xmax


def resample(data: Union[pd.DataFrame, pd.Series], target: int):
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


def find_center_mask(mask: tf.Tensor) -> tf.Tensor:
    """
    Find the centermost mask in the image.
    Args:
        mask (tf.Tensor): mask tensor

    Returns:
        tf.Tensor: new tensor with the centermost mask only
    """

    assert isinstance(mask, tf.Tensor), (
        f"Input must be a tf.Tensor, not {type(mask)}")

    if mask.dtype != bool:
        mask = tf.not_equal(mask, 0.0)

    mask_uint8 = tf.cast(mask, tf.uint8).numpy()

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask_uint8))
    mask_center = np.array(mask.shape) // 2

    distances = np.linalg.norm(centroids - mask_center, axis=1)
    closest_label = np.argmin(distances[1:]) + 1
    center_mask = (labels == closest_label).astype('uint8')

    return center_mask


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image so that pixel values are between 0 and 1,
    and it is of dtype tf.float32.
    Args:
        image (tf.Tensor): The image to normalize

    Returns:
        tf.Tensor: The normalized image
    """

    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)

    image = tf.transpose(image, perm=[1, 2, 0])

    return image / 65535.0
