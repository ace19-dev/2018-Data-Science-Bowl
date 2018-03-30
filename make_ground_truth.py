from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys

from PIL import Image
from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt


def get_image_size(data):
    image_path = os.path.join(FLAGS.dataset_dir, data, 'images')
    image = os.listdir(image_path)
    img = Image.open(os.path.join(image_path, image[0]))

    return img.height, img.width


def main(_):

    filelist = sorted(os.listdir(FLAGS.dataset_dir))
    for data in filelist:
        height, width = get_image_size(data)

        mask_path = os.path.join(FLAGS.dataset_dir, data, 'masks')
        mask_images = sorted(os.listdir(mask_path))
        mask = np.zeros((height, width, 1), dtype=np.bool)
        for mask_file in mask_images:
            _mask = imread(os.path.join(mask_path, mask_file))
            _mask = np.expand_dims(_mask, axis=-1)
            mask = np.maximum(mask, _mask)

        gt_path = os.path.join(FLAGS.ground_truth_dir, data, 'gt_mask')
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)

        # imshow(np.squeeze(mask))
        # plt.show()

        mask = np.squeeze(mask)
        img = Image.fromarray(mask)
        img.save(os.path.join(gt_path, data + '.png'))
        # img.show(title=X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--ground_truth_dir',
        default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        type=str,
        help="ground_truth data directory")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


