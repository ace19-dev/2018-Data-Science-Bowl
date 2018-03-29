import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path: str, gray: bool=False) -> np.ndarray:
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg
        gray (bool): Grayscale flag

    Returns:
        np.ndarray:
          3D numpy array of shape (H, W, 3) or 2D Grayscale Image (H, W)
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image_and_resize(image_path: str,
                          new_WH: Tuple[int, int]=(512, 512),
                          save_dir: str="resize") -> str:
    """Reads an image and resize it

    1) open `image_path` that is image.jpg
    2) resize to `new_WH`
    3) save to save_dir/image.jpg
    4) returns `image_path`

    Args:
        image_path (str): /path/to/image.jpg
        new_WH (tuple): Target width & height to resize
        save_dir (str): Directory name to save a resized image

    Returns:
        image_path (str): same as input `image_path`
    """
    assert os.path.exists(save_dir) is True
    new_path = os.path.join(save_dir, os.path.basename(image_path))
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_WH, interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, image)

    return image_path


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