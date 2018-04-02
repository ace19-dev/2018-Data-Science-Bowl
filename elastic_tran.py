# Import stuff
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.io import imread, imshow
from tqdm import tqdm


TRAIN_PATH = '../../dl_data/nuclei_dataset/stage1_train/'
TEST_PATH = '../../dl_data/nuclei_dataset/stage1_test/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    image_ = cv2.imread(path + '/images/' + id_ + '.png')
    # image_ = cv2.cvtColor(image_, cv2.COLOR_RGBA2GRAY)
    # imshow(image_)
    # plt.show()
    mask_ = cv2.imread(path + '/gt_mask/' + id_ + '.png')
    # imshow(mask_)
    # plt.show()

    image = np.concatenate((image_, mask_), axis=2)

    alpha = image.shape[1] * 2
    sigma = image.shape[1] * 0.08
    alpha_affine = image.shape[1] * 0.05
    random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))


    im_merge_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    # im_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    # mask_t = map_coordinates(mask_, indices, order=1, mode='reflect').reshape(shape)

    im_t = im_merge_t[...,0:3]
    # imshow(im_t)
    # plt.show()
    mask_t = im_merge_t[...,3:6]
    mask_t = cv2.cvtColor(mask_t, cv2.COLOR_RGB2GRAY)
    # imshow(mask_t)
    # plt.show()

    new_id = 'elastic' + id_[7:]
    os.mkdir(TRAIN_PATH + new_id)
    os.mkdir(TRAIN_PATH + new_id + '/images/')
    os.mkdir(TRAIN_PATH + new_id + '/gt_mask/')
    cv2.imwrite(TRAIN_PATH + new_id + '/images/' + new_id + '.png', im_t)
    cv2.imwrite(TRAIN_PATH + new_id + '/gt_mask/' + new_id + '.png', mask_t)

"""
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = cv2.imread(path + '/masks/' + mask_file, -1)
        # change from GRAY to RGBA
        mask_ = cv2.cvtColor(mask_, cv2.COLOR_GRAY2RGBA)
        mask_t = map_coordinates(mask_, indices, order=1, mode='reflect').reshape(shape)
        mask_t = cv2.cvtColor(mask_t, cv2.COLOR_RGBA2GRAY)
        new_mask_id = 'elastic' + mask_file[7:]
        cv2.imwrite(TRAIN_PATH + new_id + '/masks/' + new_mask_id + '.png', mask_t)
"""
