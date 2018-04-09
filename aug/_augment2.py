import os
import sys
import argparse
import tqdm
import uuid

import cv2  # To read and manipulate images
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.image_utils import read_image, read_mask
from tqdm import tqdm


# RANDOM_SEED = 777

FLAGS = None

def read_images_and_gt_masks () :
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    x_train = []
    masks = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = FLAGS.train_dir + id_
        # image_ = cv2.imread(path + '/images/' + id_ + '.png')
        image_ = read_image(path + '/images/' + id_ + '.png', target_size=(256, 256))
        mask_ = read_image(path + '/gt_mask/' + id_ + '.png', target_size=(256, 256))

        x_train.append(image_)
        masks.append(mask_)

    x_train = np.array(x_train)
    masks = np.array(masks)

    return x_train, masks

def read_train_data_properties(train_dir):
    """Read basic properties of training images and masks"""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(train_dir))[1]):
        img_dir = os.path.join(train_dir, dir_name, 'images')
        mask_dir = os.path.join(train_dir, dir_name, 'gt_mask')
        img_name = next(os.walk(img_dir))[2][0]
        mask_name = next(os.walk(mask_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1], img_shape[2],
                    img_path, img_dir, mask_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width', 'num_channels',
                                          'image_path', 'image_dir', 'mask_path', 'mask_dir'])
    return train_df


def read_test_data_properties(test_dir):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, 'images')
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels', 'image_path'])
    return test_df


def load_raw_data(train_df, test_df, target_size=None):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, x_test = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=target_size)
        mask = read_mask(train_df['mask_dir'].loc[i], target_size=target_size)
        x_train.append(img)
        y_train.append(mask)

    # Read and resize test images.
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=target_size)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    y_train = np.expand_dims(np.array(y_train), axis=4)
    x_test = np.array(x_test)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))

    return x_train, y_train, x_test


def normalize(data, type_=1):
    """Normalize data."""
    if type_ == 0:
        # Convert pixel values from [0:255] to [0:1] by global factor
        data = data.astype(np.float32) / data.max()
    if type_ == 1:
        # Convert pixel values from [0:255] to [0:1] by local factor
        div = data.max(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        div[div < 0.01 * data.mean()] = 1.  # protect against too small pixel intensities
        data = data.astype(np.float32) / div
    if type_ == 2:
        # Standardisation of each image
        data = data.astype(np.float32) / data.max()
        mean = data.mean(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        std = data.std(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        data = (data - mean) / std

    return data


def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)


def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(list(map(lambda x: 1. - x if np.mean(x) > cutoff else x, imgs)))
    return normalize_imgs(imgs)


def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs


# Normalize all images and masks. There is the possibility to transform images
# into the grayscale sepctrum and to invert images which have a very
# light background.
def preprocess_raw_data(x_train, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        print('Images inverted to remove light backgrounds.')

    return x_train


def write_image(x_train, masks):
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        image_ = x_train[n, :, :, :]
        mask_ = masks[n, :, :, :]
        randomString = str(uuid.uuid4()).replace("-", "")

        new_id = FLAGS.aug_prefix + randomString + id_[39:]
        os.mkdir(FLAGS.train_dir + new_id)
        os.mkdir(FLAGS.train_dir + new_id + '/images/')
        os.mkdir(FLAGS.train_dir + new_id + '/gt_mask/')
        cv2.imwrite(FLAGS.train_dir + new_id + '/images/' + new_id + '.png', image_)
        cv2.imwrite(FLAGS.train_dir + new_id + '/gt_mask/' + new_id + '.png', mask_)


def image_augmentation(image, mask):
    """Returns (maybe) augmented images
    (1) Random flip (left <--> right)
    (2) Random flip (up <--> down)
    (3) Random brightness
    (4) Random hue
    Args:
        image (3-D Tensor): Image tensor of (H, W, C)
        mask (3-D Tensor): Mask image tensor of (H, W, 1)
    Returns:
        image: Maybe augmented image (same shape as input `image`)
        mask: Maybe augmented mask (same shape as input `mask`)
    """
    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)

    return image, mask


def make_aug_dir():
    randomString = str(uuid.uuid4()).replace("-", "")
    _new = FLAGS.aug_prefix + randomString

    return _new


def main(_):
    train_info = read_train_data_properties(FLAGS.train_dir)
    test_info = read_test_data_properties(FLAGS.test_dir)

    seed = np.random.randint(10000)

    # x_train, y_train, x_test = load_raw_data(train_info, test_info)
    #x_train, y_train, x_test = \
    #    preprocess_raw_data(x_train, y_train, x_test, invert=True)

    x_train, y_train = read_images_and_gt_masks()
    x_train = preprocess_raw_data(x_train, grayscale=False, invert=True)

    write_image(x_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default='../../../dl_data/nucleus/stage1_train/',
        type=str,
        help="Train Data directory")

    # parser.add_argument(
    #     '--aug_dir',
    #     default='../../../dl_data/nucleus/aug_stage1_train',
    #     type=str,
    #     help="Augmentation train Data directory")

    parser.add_argument(
        '--test_dir',
        default='../../../dl_data/nucleus/stage1_test',
        type=str,
        help="Test data directory")

    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help="Image height and width")

    parser.add_argument(
        '--aug_prefix',
        default='_aug_',
        type=str,
        help="prefix name of augmentation")

    parser.add_argument(
        '--aug_count',
        type=int,
        default=2,
        help="Count of augmentation")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)