import cv2
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import tifffile as tiff

from pathlib import Path
from tqdm import tqdm

from . import helpers as h


def _preprocess_wayne_rpe(raw_labels: str,
                          raw_images: str,
                          data_dir: str,
                          labels: str,
                          crop_size: int,
                          drop_na: bool = True,
                          dynamic_crop: bool = False,
                          **kwargs) -> None:
    """
    Preprocessing for wayne rpe dataset
    """

    raw_images = Path(raw_images)
    data_dir = Path(data_dir)

    print('Preprocessing Wayne Datset...')
    print(f'Dynamic Crop: {dynamic_crop}')
    orig_labels = pd.read_csv(raw_labels)

    if drop_na:
        print(f'Original number of cells in dataset {orig_labels.shape[0]}')
        labels_proc = orig_labels.dropna()

        print(f'Afer dropping NA values: {labels_proc.shape[0]}')

    else:
        labels_proc = orig_labels.copy()

    # Convert cell id to match image file names
    labels_proc["cell_id"] = (
        labels_proc["cell_id"].apply(lambda x: f"cell_{x:04d}"))

    # Remove M phase cells
    labels_proc = labels_proc[labels_proc['pred_phase'] != "M"]

    # One-hot encode classes
    labels_onehot = pd.get_dummies(labels_proc['pred_phase']).astype(int)
    labels_proc = pd.concat([labels_proc, labels_onehot], axis=1)

    # Add file paths to the csv
    labels_proc['filepath'] = labels_proc['cell_id'].apply(
        lambda x: str(data_dir / f'{x}.npy')
    )

    data_dir.mkdir(exist_ok=True, parents=True)
    Path(labels).parent.mkdir(exist_ok=True, parents=True)

    # Save labels
    labels_proc.to_csv(labels, index=False)

    print('Processing and saving image masks.')
    for cell_id in tqdm(labels_proc['cell_id'], desc="Processing",
                        unit="images", ncols=100):

        img_path = raw_images / f"{cell_id}.tif"

        with tiff.TiffFile(img_path) as tif:
            image = tif.asarray()

        image = normalize_image(image)

        nuc_mask = image[:, :, 57]
        ring_mask = image[:, :, 58]
        combined_mask = tf.maximum(nuc_mask, ring_mask)

        masks = [nuc_mask, ring_mask, combined_mask]
        centered_masks = [find_center_mask(mask) for mask in masks]
        masks.extend(centered_masks)

        masks_to_add = tf.cast(tf.stack(masks[2:], axis=-1), tf.float32)
        image = tf.concat([image, masks_to_add], axis=-1)

        image_np = image.numpy()
        save_path = data_dir / f'{cell_id}.npy'

        np.save(save_path, image_np)

        ymin, ymax, xmin, xmax = get_min_max_axis(image[:, :, -1])

        offset_height = max(ymin - 5, 0)
        offset_width = max(xmin - 5, 0)

        max_height = 0
        max_width = 0
        ymin, ymax, xmin, xmax = get_min_max_axis(image[:, :, -1])

        offset_height = max(ymin - 5, 0)
        offset_width = max(xmin - 5, 0)
        target_height = min((ymax + 5) - offset_height, image.shape[0])
        target_width = min((xmax + 5) - offset_width, image.shape[1])

        if target_height > max_height:
            max_height = target_height

        if target_width > max_width:
            max_width = target_width

    if dynamic_crop:
        print(f'Found maximum dimensions:'
              f'\nHeight:\t{max_height}\nWidth:\t{max_width}')

        hw = max(max_height, max_width)

        if crop_size:
            if hw > crop_size:
                warn = '\n\nWARNING: the max size of the cells in this ' + \
                    'dataset isgreater than the specified crop size. ' + \
                    'Crop anyway?\n' + \
                    f'Necessary Crop Size:\t{hw}x{hw}\n' + \
                    f'Specified Crop Size:\t{crop_size}x{crop_size}\n'

                cont = h.yes_no(warn)

                if not cont:
                    sys.exit(1)

            hw = crop_size

        print(f'Cropping images to: {hw}x{hw}')
        multi_cell = 0
        for cell_id in tqdm(labels_proc['cell_id'], desc="Processing",
                            unit="images", ncols=100):
            img_path = str(data_dir / f'{cell_id}.npy')
            image = np.load(img_path)

            cropped_image = tf.image.crop_to_bounding_box(
                image, offset_height, offset_width, hw, hw)

            np.save(img_path, cropped_image)

            unique_values, _ = tf.unique(
                tf.reshape(cropped_image[:, :, 59], [-1]))

            if unique_values.shape[0] > 2:
                multi_cell += 1

        print('Finished cropping images.')

        if multi_cell != 0:
            print(
                f'\n{multi_cell}/{labels_proc.shape[0]} '
                f'({(multi_cell / labels_proc.shape[0])*100:.2f}%) '
                'images still have more than one cell in them.')

        else:
            print('Successfully eliminated extra cells from all images.')

    print('Finished preprocessing dataset.')


def preprocess(dataset_name: str, **kwargs):
    if dataset_name.lower() == 'wayne':
        _preprocess_wayne_rpe(**kwargs)

    elif 'wayne_crop' in dataset_name.lower():
        if 'inference' in dataset_name.lower():
            kwargs['labels'] = kwargs['data_csv']
            kwargs['drop_na'] = False

        _preprocess_wayne_rpe(dynamic_crop=True, **kwargs)


def normalize_image(image: np.ndarray) -> tf.Tensor:
    """
    Normalize the image so that pixel values are between 0 and 1,
    and it is of dtype tf.float32.
    Args:
        image (tf.Tensor): The image to normalize

    Returns:
        tf.Tensor: The normalized image
    """

    image = image.astype(np.float32)

    image /= 65535.0

    image = np.transpose(image, (1, 2, 0))

    image_tf = tf.convert_to_tensor(image)

    return image_tf


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


def get_min_max_axis(img: tf.Tensor):
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

    xmin = tf.reduce_min(xidx).numpy()
    ymin = tf.reduce_min(yidx).numpy()

    xmax = tf.reduce_max(xidx).numpy()
    ymax = tf.reduce_max(yidx).numpy()

    return ymin, ymax, xmin, xmax
