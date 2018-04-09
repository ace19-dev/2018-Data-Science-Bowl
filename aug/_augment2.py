import os
import sys
import argparse
import tqdm
import uuid

import cv2  # To read and manipulate images
import numpy as np
import pandas as pd
import tensorflow as tf

# For using image generation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from utils.oper_utils2 \
    import normalize_imgs, trsf_proba_to_binary, \
            normalize_masks, imgs_to_grayscale, invert_imgs

from utils.image_utils import read_image, read_mask


# RANDOM_SEED = 777

FLAGS = None


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


# Normalize all images and masks. There is the possibility to transform images
# into the grayscale sepctrum and to invert images which have a very
# light background.
def preprocess_raw_data(x_train, y_train, x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    y_train = trsf_proba_to_binary(normalize_masks(y_train))
    x_test = normalize_imgs(x_test)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return x_train, y_train, x_test


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


# def get_image_mask(queue, augmentation=True):
#     text_reader = tf.TextLineReader(skip_header_lines=1)
#     _, csv_content = text_reader.read(queue)
#
#     image_path, mask_path = tf.decode_csv(
#         csv_content, record_defaults=[[""], [""]])
#
#     image_file = tf.read_file(image_path)
#     mask_file = tf.read_file(mask_path)
#
#     image = tf.image.decode_jpeg(image_file, channels=3)
#     image.set_shape([height, width, 3])
#     image = tf.cast(image, tf.float32)
#
#     mask = tf.image.decode_jpeg(mask_file, channels=1)
#     mask.set_shape([height, width, 1])
#     mask = tf.cast(mask, tf.float32)
#     mask = mask / (tf.reduce_max(mask) + 1e-7)
#
#     if augmentation:
#         image, mask = image_augmentation(image, mask)
#
#     return image, mask


def main(_):
    train_info = read_train_data_properties(FLAGS.train_dir)
    test_info = read_test_data_properties(FLAGS.test_dir)

    seed = np.random.randint(10000)

    x_train, y_train, x_test = load_raw_data(train_info, test_info)
    x_train, y_train, x_test = \
        preprocess_raw_data(x_train, y_train, x_test, invert=True)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default='../../../dl_data/nucleus/stage1_train',
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