import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf



def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)
    return tf.expand_dims(mask, -1)


def plot_image(image: np.ndarray, title: Optional[str]=None, **kwargs) -> None:
    """Plot a single image

    Args:
        image (2-D or 3-D array): image as a numpy array (H, W) or (H, W, C)
        title (str, optional): title for a plot
        **kwargs: keyword arguemtns for `plt.imshow`
    """
    shape = image.shape

    if len(shape) == 3:
        plt.imshow(image, **kwargs)
    elif len(shape) == 2:
        plt.imshow(image, **kwargs)
    else:
        raise TypeError(
            "2-D array or 3-D array should be given but {} was given".format(
                shape))

    if title:
        plt.title(title)

    plt.show()