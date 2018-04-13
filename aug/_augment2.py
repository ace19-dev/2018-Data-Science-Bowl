import os
import sys
import argparse
import tqdm
import uuid

import cv2  # To read and manipulate images
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.oper_utils2 \
    import normalize_imgs, trsf_proba_to_binary, \
            normalize_masks, imgs_to_grayscale, invert_imgs

from utils.image_utils import read_image, read_mask
from tqdm import tqdm

RANDOM_SEED = 54989

FLAGS = None


def read_images_and_gt_masks () :
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    images = []
    masks = []

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = FLAGS.train_dir + id_
        image_ = read_image(path + '/images/' + id_ + '.png',
                            target_size=(FLAGS.img_size, FLAGS.img_size))
        # mask_ = read_mask(path + '/masks/',
        #                   target_size=(FLAGS.img_size, FLAGS.img_size))
        mask_ = read_image(path + '/gt_mask/' + id_ + '.png',
                           color_mode=cv2.IMREAD_GRAYSCALE,
                           target_size=(FLAGS.img_size, FLAGS.img_size))

        images.append(image_)
        masks.append(tf.expand_dims(mask_, -1))

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


# def read_test () :
#     test_ids = next(os.walk(FLAGS.test_dir))[1]
#     x_test = []
#
#     for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#         path = FLAGS.test_dir + id_
#         image_ = read_image(path + '/images/' + id_ + '.png',
#                             target_size=(FLAGS.img_size, FLAGS.img_size))
#
#         x_test.append(image_)
#
#     x_test = np.array(x_test)
#
#     return x_test


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
    # return imgs


def imgs_to_grayscale(imgs):
    # imgs = tf.reshape(imgs, [-1, 256, 256, 3])
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        # imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
        imgs = np.expand_dims(np.mean(imgs, axis=3), axis=3)
    return imgs


# Normalize all images and masks. There is the possibility to transform images
# into the grayscale sepctrum and to invert images which have a very
# light background.
def preprocess_raw_data(x_train, grayscale=False, invert=False):
    """Preprocessing of images and masks."""

    # Normalize images and masks
    # x_train = normalize_imgs(x_train)
    # print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        print('Images inverted to remove light backgrounds.')

    return x_train


def write_image(images, masks):
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        image_ = images[n, :, :, :]
        mask_ = masks[n, :, :, :]

        randomString = str(uuid.uuid4()).replace("-", "")
        new_id = FLAGS.aug_prefix + id_[:10] + randomString

        os.mkdir(FLAGS.train_dir + new_id)
        os.mkdir(FLAGS.train_dir + new_id + '/images/')
        os.mkdir(FLAGS.train_dir + new_id + '/gt_mask/')
        cv2.imwrite(FLAGS.train_dir + new_id + '/images/' + new_id + '.png', image_)
        cv2.imwrite(FLAGS.train_dir + new_id + '/gt_mask/' + new_id + '.png', mask_)

def write_image2(images, masks):
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        image_ = images[n, :, :, :]
        mask_ = masks[n, :, :, :]

        randomString = str(uuid.uuid4()).replace("-", "")
        new_id = FLAGS.aug_prefix + id_[:10] + randomString

        os.mkdir(FLAGS.train_dir + new_id)
        os.mkdir(FLAGS.train_dir + new_id + '/images/')
        os.mkdir(FLAGS.train_dir + new_id + '/gt_mask/')
        cv2.imwrite(FLAGS.train_dir + new_id + '/images/' + new_id + '.png', image_)
        cv2.imwrite(FLAGS.train_dir + new_id + '/gt_mask/' + new_id + '.png', mask_)


def image_augmentation(image, seed):
    """Returns (maybe) augmented images`
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
    maybe_flipped = tf.image.random_flip_left_right(image, seed=seed)
    maybe_flipped = tf.image.random_flip_up_down(image, seed=seed)

    if image.shape[2] == 3:
        image = tf.image.random_brightness(image, 0.7, seed=seed)
        image = tf.image.random_hue(image, 0.3, seed=seed)

    return image


# def make_aug_dir():
#     randomString = str(uuid.uuid4()).replace("-", "")
#     _new = FLAGS.aug_prefix + randomString
#
#     return _new


def main(_):
    images, masks = read_images_and_gt_masks()
    # images = preprocess_raw_data(images, grayscale=True, invert=False)

    image_list = []
    mask_list = []
    for idx, img in enumerate(images):
        seed = np.random.randint(RANDOM_SEED)

        _image = image_augmentation(images[idx], seed)
        _mask = image_augmentation(masks[idx], seed)

        image_list.append(_image)
        mask_list.append(_mask)

    images = np.array(image_list)
    masks = np.array(mask_list)

    write_image(images, masks)

    # x_test = read_test()
    # x_test = imgs_to_grayscale(x_test)


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