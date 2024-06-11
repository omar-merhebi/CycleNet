import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from scipy import ndimage
from typing import Union, Tuple


class WayneCroppedDataset(tf.keras.utils.Sequence):
    def __init__(self, data_idx, shuffle, balance, batch_size,
                 data_dir, labels, channels, augment, mask, fill,
                 log_image, **kwargs):

        self.data_idx = data_idx
        self.shuffle = shuffle
        self.balance = balance
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.augment = augment
        self.mask = mask
        self.fill = fill
        self.log_image = log_image

        labels = pd.read_csv(labels)
        labels = labels.loc[data_idx]

        if self.balance:
            val_counts = labels['pred_phase'].value_counts()
            if self.balance.lower() == 'middle':
                # For now we just hard code:
                sample_to = val_counts['S']

            elif self.balance.lower() == 'up':
                sample_to = val_counts['G1']

            else:
                sample_to = val_counts['G2']

            labels_g1 = resample(
                labels[labels['G1'] == 1],
                target=sample_to
            )

            labels_s = resample(
                labels[labels['S'] == 1],
                target=sample_to
            )

            labels_g2 = resample(
                labels[labels['G2'] == 1],
                target=sample_to
            )

            labels = pd.concat([labels_g1, labels_g2, labels_s])

        self.labels = labels

        self.indexes = np.arange(labels.shape[0])

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, idx):
        idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_imgs = [self._load_img(i) for i in idx]

        log_imgs = [img[-1] for img in batch_imgs]
        batch_imgs = [img[0] for img in batch_imgs]

        batch_labels = [tf.convert_to_tensor(
            self.labels[['G1', 'S', 'G2']].iloc[i].values) for i in idx]

        X = tf.stack(batch_imgs)
        lab = tf.stack(batch_labels)
        log_imgs = tf.stack(log_imgs)

        # self.log_imgs = log_imgs[:, :, :5]

        return X, lab

    def _load_img(self, idx):
        filepath = self.labels['filepath'].iloc[idx]
        image = np.load(filepath)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        inv_mask = tf.cast(tf.not_equal(image[:, :, 59], 0), dtype=tf.float32)
        inv_mask = 1 - inv_mask

        # stats = [self._extracellular_stats(image[:, :, slc], inv_mask)
        #          for slc in range(image.shape[-1])]

        if self.augment:
            image = self._augment(image)

        if self.mask:
            if self.mask.lower() == 'nuc':
                mask_id = 60

            else:
                mask_id = 62

            mask = image[:, :, mask_id]
            mask = tf.cast(tf.not_equal(mask, 0), dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, [1, 1, image.shape[-1]])

            image *= tf.cast(mask, image.dtype)

        if self.fill:
            filled = []
            for i in range(image.shape[-1]):
                slice_2d = image[:, :, i]
                mean, stddev = stats[i]
                filled_slice = self._fill_zeros(slice_2d, mean, stddev)
                filled.append(filled_slice)

            image = tf.stack(filled, axis=-1)

        log_image = image[:, :, self.log_image]

        image = tf.gather(image, self.channels, axis=-1)

        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)

        return image, log_image

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
            return image

        random_vals = tf.random.normal([tf.shape(zeros)[0]],
                                       mean=mean, stddev=stddev,
                                       dtype=image.dtype)

        filled = tf.tensor_scatter_nd_update(image, zeros, random_vals)

        return filled

    def _augment(self, image):
        angle = np.random.randint(0, 360)
        image_np = image.numpy()

        image_np_rot = ndimage.rotate(image_np, angle, axes=[0, 1],
                                      reshape=False, order=0)

        image = tf.convert_to_tensor(image_np_rot, dtype=tf.float32)

        ymin, ymax, xmin, xmax = get_min_max_axis(image[:, :, -1])

        yshift = np.random.randint(low=ymin, high=ymax)
        xshift = np.random.randint(low=xmin, high=xmax)

        image = tf.roll(image, shift=[yshift, xshift], axis=[0, 1])

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

        if tf.size(vals) == 0:
            return 0.0, 0.0

        mean = tf.math.reduce_mean(vals)
        stddev = tf.math.reduce_std(vals)

        return mean, stddev

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


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
