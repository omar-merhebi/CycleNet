import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

from pathlib import Path
from tqdm import tqdm


def _preprocess_wayne_rpe(raw_labels: str,
                          raw_images: str,
                          data_dir: str,
                          labels: str,
                          dynamic_crop: bool = False, **kwargs) -> None:
    """
    Preprocessing for wayne rpe dataset
    """

    raw_images = Path(raw_images)
    data_dir = Path(data_dir)

    print('Preprocessing Wayne Datset...')
    print(f'Dynamic Crop: {dynamic_crop}')
    orig_labels = pd.read_csv(raw_labels)

    print(f'Original number of cells in dataset {orig_labels.shape[0]}')
    labels_proc = orig_labels.dropna()

    print(f'Afer dropping NA values: {labels_proc.shape[0]}')

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

        nuc_mask = image[57]
        ring_mask = image[58]
        combined_mask = np.maximum(nuc_mask, ring_mask)

        masks = [nuc_mask, ring_mask, combined_mask]
        centered_masks = [find_center_mask(mask) for mask in masks]
        masks.extend(centered_masks)

        masks_to_add = np.stack(masks[2:], axis=0).astype(np.float32)
        image = np.concatenate([image, masks_to_add], axis=0)

        save_path = data_dir / f'{cell_id}.npy'

        np.save(save_path, image)

        ymin, ymax, xmin, xmax = get_min_max_axis(image[-1])

        offset_height = max(ymin - 5, 0)
        offset_width = max(xmin - 5, 0)

        max_height = 0
        max_width = 0
        ymin, ymax, xmin, xmax = get_min_max_axis(image[-1])

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

        print(f'Cropping images to: {hw}x{hw}')
        multi_cell = 0
        for cell_id in tqdm(labels_proc['cell_id'], desc="Processing",
                            unit="images", ncols=100):
            img_path = str(data_dir / f'{cell_id}.npy')
            image = np.load(img_path)

            cropped_image = image[:, offset_height:offset_height+hw,
                                  offset_width:offset_width+hw]

            np.save(img_path, cropped_image)

            unique_values = np.unique(cropped_image[-4])

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

    elif dataset_name.lower() == 'wayne_crop':
        _preprocess_wayne_rpe(dynamic_crop=True, **kwargs)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image so that pixel values are between 0 and 1,
    and it is of dtype np.float32.
    Args:
        image (np.ndarray): The image to normalize

    Returns:
        np.ndarray: The normalized image
    """

    image = image.astype(np.float32)

    image /= 65535.0

    return image


def find_center_mask(mask: np.ndarray) -> np.ndarray:
    """
    Find the centermost mask in the image.
    Args:
        mask (np.ndarray): mask tensor

    Returns:
        np.ndarray: new tensor with the centermost mask only
    """

    assert isinstance(mask, np.ndarray), (
        f"Input must be a NumPy array, not {type(mask)}")

    if mask.dtype != bool:
        mask = np.not_equal(mask, 0.0)

    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask_uint8))
    mask_center = np.array(mask.shape) // 2

    distances = np.linalg.norm(centroids - mask_center, axis=1)
    closest_label = np.argmin(distances[1:]) + 1
    center_mask = (labels == closest_label).astype('uint8')

    return center_mask


def get_min_max_axis(img: np.ndarray):
    """
    Finds the extreme points of a binary mask along the x and y axes.
    Args:
        img (np.ndarray): the binary mask as a torch.Tensor

    Returns:
        tuple: tuple of the minimum and maximum values along the x and y axes
               (ymin, ymax, xmin, xmax)
    """

    assert isinstance(img, np.ndarray), (
        f"Input must be a NumPy array, not {type(img)}")

    nonzero = np.argwhere(img != 0)

    xidx = np.unique(nonzero[:, 1])
    yidx = np.unique(nonzero[:, 0])

    xmin = np.min(xidx)
    ymin = np.min(yidx)

    xmax = np.max(xidx)
    ymax = np.max(yidx)

    return ymin, ymax, xmin, xmax
