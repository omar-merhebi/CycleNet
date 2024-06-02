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

logging.basicConfig(level=logging.INFO)


class WayneRPEDataset(tf.keras.utils.Sequence):
    def __init__(self, cfg: DictConfig, data_idx: np.ndarray,
                 augment: bool = False):
        self.cfg = cfg
        self.data_idx = data_idx
        self.augment = augment
        self.shuffle = cfg.mode.data_shuffle
        self.batch_size = cfg.mode.batch_size

        channel_annotations = pd.read_csv(cfg.dataset.channel_annotations)
        channel_annotations['feature'] = (
            channel_annotations['feature'].apply(lambda x: x.lower().strip()))

        self.channel_annotations = channel_annotations

        self.channels = (
            [channel.lower().strip()
             for channel in self.cfg.dataset.channels]
            if self.cfg.dataset.channels else None)

        self.log_image = [self.cfg.dataset.log_image.lower().strip()]
        self.input_channels = (len(self.channels)
                               if self.channels else 55)

        labels = pd.read_csv(cfg.dataset.labels)
        labels['phase_index'], self.unique_phases = (
            pd.factorize(labels['pred_phase']))

        labels = labels.iloc[self.data_idx].reset_index(drop=True)

        if self.cfg.dataset.balancing:
            balancing = self.cfg.dataset.balancing.lower()
            class_counts = Counter(labels['pred_phase'])
            if 'over' in balancing or 'up' in balancing:
                target = class_counts['G1']

            elif 'under' in balancing or 'down' in balancing:
                target = class_counts['G2']

            else:
                target = class_counts['S']

            labels_g1 = resample(labels[labels['pred_phase'] == 'G1'], target)
            labels_s = resample(labels[labels['pred_phase'] == 'S'], target)
            labels_g2 = resample(labels[labels['pred_phase'] == 'G2'], target)

            labels = pd.concat([labels_g1, labels_s, labels_g2])

            labels_onehot = pd.get_dummies(labels['pred_phase'])

            self.n_classes = labels_onehot.shape[1]

            labels = labels.join(labels_onehot)

            self.labels = labels

            self.indexes = np.arange(labels.shape[0])

            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return self.labels.shape[0] // self.cfg.mode.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx *
                               self.batch_size:(idx + 1) *
                               self.batch_size]

        # Pre-allocate batch tensors
        batch_images = np.zeros((self.batch_size,
                                 *self._data_generation(indexes[0])[0].shape),
                                dtype=np.float32)
        batch_phase_lab = np.zeros((self.batch_size, self.n_classes),
                                   dtype=np.float32)
        batch_age_lab = np.zeros((self.batch_size, ), dtype=np.float32)
        batch_cell_id = []

        if self.cfg.dataset.log_image:
            log_image_shape = self._data_generation(indexes[0])[3].shape
            batch_log_images = np.zeros((self.batch_size, *log_image_shape),
                                        dtype=np.float32)

        else:
            batch_log_images = None

        for i, ind in enumerate(indexes):
            image, phase_lab, age_lab, log_image, cell_id = (
                self._data_generation(ind))

            batch_images[i] = image
            batch_phase_lab[i] = phase_lab
            batch_age_lab[i] = age_lab
            batch_cell_id.append(cell_id)
            if self.cfg.dataset.log_image:
                batch_log_images[i] = log_image

        # Convert to tensors
        batch_images = tf.convert_to_tensor(batch_images)
        batch_phase_lab = tf.convert_to_tensor(batch_phase_lab)
        batch_age_lab = tf.convert_to_tensor(batch_age_lab)

        if self.cfg.dataset.log_image:
            batch_log_images = tf.convert_to_tensor(batch_log_images)

        return (batch_images, batch_phase_lab,
                batch_age_lab, batch_log_images,
                batch_cell_id)

    def _data_generation(self, idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns the data tensor from the dataset as well as a log image
        (as defined in config).
        Args:
            idx (int): index of data to obtain
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: the image to use and the log image
        """
        image_pth = (
            self.cfg.dataset.data_dir /
            f'{self.labels.iloc[idx]["cell_id"]}.tif')

        phase_lab = self.labels.iloc[idx][["G1", "S", "G2"]].values
        age_lab = self.labels.iloc[idx]["pred_age"]
        cell_id = self.labels.iloc[idx]["cell_id"]

        image = tiff.imread(image_pth)
        image = normalize_image(image)

        # add masks
        image = self._append_masks(image)

        # get median and stdv of each dim of the image
        outmask = 1 - image[:, :, 59]
        stats = [self._extracellular_stats(image[:, :, slc], outmask)
                 for slc in range(image.shape[-1])]

        # Apply masks if enabled
        if self.cfg.dataset.mask:
            mask_id = self.cfg.dataset.mask.lower()

            if mask_id == 'nuc':
                mask_idx = 60

            else:
                mask_idx = 62

            mask = image[:, :, mask_idx]
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, [1, 1, image.shape[-1]])

            image = image * tf.cast(mask, image.dtype)

        if self.cfg.dataset.augment:
            image = self._augment(image, mask=image[:, :, 62])

        if self.cfg.dataset.log_image:
            channel_idx = list(
                set(self.channel_annotations[
                    self.channel_annotations['feature'].isin(self.log_image)
                ]['frame'].tolist()))

            assert len(channel_idx) == 1, (
                'Can only log one image to wandb,'
                f'found {len(channel_idx)} images')

            channel_idx = int(channel_idx[0])

            log_image = image[:, :, channel_idx]

        else:
            log_image = None

        if self.cfg.dataset.channels:
            use_channels = self.cfg.dataset.channels.lower()
            channel_idx = list(
                set(self.channel_annotations[
                    self.channel_annotations['feature'].isin(self.log_image)
                ]['frame'].tolist()))

            channel_idx = tf.constant(channel_idx, dtype=tf.int32)

            assert len(use_channels) == self.input_channels, (
                'The names of one or more channels provided'
                'dot not match those found in the annotation file.'
            )

            image = tf.gather(image, channel_idx, axis=-1)

        else:
            # Drop all masks and use the first 55 slices
            image = image[:, :, :55]

        if self.cfg.dataset.fill.enabled:
            filled = []
            for i in range(image.shape[-1]):
                slice_2d = image[:, :, i]
                mean, stddev = stats[i]
                filled_slice = self._fill_zeros(slice_2d, mean, stddev)
                filled.append(filled_slice)

            image = tf.stack(filled, axis=-1)

        return image, phase_lab, age_lab, log_image, cell_id

    def _fill_zeros(self, image: tf.Tensor,
                    mean: float, stddev: float) -> tf.Tensor:
        """
        Fills in zero values provided in an image with values randomly sampled
        from a Gaussian distrubition with the mean and stddev provided.
        Args:
            image (tf.Tensor): 2D image tensor with zero pixels to fill.
            mean (float): Mean pixel value for the random sample.
            stddev (float): Standard deviation of pixel value for the
            random sample

        Returns:
            tf.Tensor: 2D image tensor with zero values filled in
        """

        assert len(image.shape) == 2, (
            'Shape of the image and mask must be 2D.'
            f'Current shape:{image.shape}'
        )

        zeros = tf.where(tf.equal(image, 0))

        if tf.size(zeros) == 0:
            logging.warning("Filling is enabled but no zero pixels found.")
            return image

        random_vals = tf.random.normal([tf.shape(zeros)[0]],
                                       mean=mean, stddev=stddev,
                                       dtype=image.dtype)

        filled = tf.tensor_scatter_nd_update(image, zeros, random_vals)

        return filled

    def _append_masks(self, image: tf.Tensor,
                      nuc_mask_idx: int = 57,
                      ring_mask_idx: int = 58) -> tf.Tensor:
        """
        Combines nucleus and ring masks and centers them and adds these 4
        additional masks: combined+centered_nuc+centered_ring+centered_combined
        to the end of the tensorflow tensor
        Args:
            image (tf.Tensor): image tensor.
            nuc_mask_idx (int, optional): index of the slice containing the
            nucleus mask. Defaults to 57.
            ring_mask_idx (int, optional): index of the slice containing the
            ring mask. Defaults to 58.

        Returns:
            tf.Tensor: tensor with new masks added to the end of dim=-1
        """

        nuc = tf.not_equal(image[:, :, nuc_mask_idx], 0.0)
        ring = tf.not_equal(image[:, :, ring_mask_idx], 0.0)
        combined = tf.logical_or(nuc, ring)

        orig_masks = [nuc, ring, combined]
        centered_masks = [find_center_mask(mask) for mask in orig_masks]

        all_masks = orig_masks + centered_masks

        masks_to_add = tf.cast(tf.stack(all_masks[2:], axis=-1), tf.float32)
        image = tf.concat([image, masks_to_add], axis=-1)

        return image

    def _extracellular_stats(self, image: tf.Tensor,
                             mask: tf.Tensor) -> Tuple[float, float]:
        """
        Get the mean and stddev of pixel values from a 2D image.
        Args:
            image (tf.Tensor): 2D image input (a slice of the image tensor).
            mask (tf.Tensor): 2D mask to mask out pixels inside cells.

        Returns:
            (mean, stddev): mean and standard deviation of the pixel values
                            of the image from outside any cells present.
        """

        assert image.shape == mask.shape, (
            'The input image and mask must be the same shape to find stats.\n'
            f'Image Shape:\t{image.shape}\nMask Shape:\t{mask.shape}')

        assert len(image.shape) == 2, (
            'Shape of the image and mask must be 2D.'
            f'Current shape:{image.shape}'
        )

        # Exclude cell and zero values
        masked_image = image * tf.cast(mask, image.dtype)
        vals = tf.boolean_mask(masked_image, masked_image != 0)

        # Get mean and stdv
        mean = tf.math.reduce_mean(vals)
        stddev = tf.math.reduce_std(vals)

        return mean, stddev

    def _augment(self, image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Method for random rotation and translation of the image
        Args:
            image (tf.Tensor): the input image to augment.
            mask (tf.Tensor): a mask to ensure the augmentation
            doesn't cut off the cell.

        Returns:
            tf.Tensor: augmented image
        """

        angle = np.random.randint(0, 360)
        image_np = image.numpy()

        image_np_rot = ndimage.rotate(image_np, angle, axes=[0, 1],
                                      reshape=False, order=0)

        image = tf.convert_to_tensor(image_np_rot, dtype=tf.float32)

        ymin, ymax, xmin, xmax = get_min_max_axis(mask)

        xshift = np.random.randint(low=xmin, high=xmax)
        yshift = np.random.randint(low=ymin, high=ymax)

        image = tf.roll(image, shift=[yshift, xshift], axis=[0, 1])

        return image


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
