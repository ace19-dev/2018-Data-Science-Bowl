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
    gt_mask_list = FLAGS.ground_truth_prefix.split(',')
    if len(gt_mask_list) != 2:
        raise Exception(
            '--ground_truth_prefix must has 2 items (with split ,)')

    equal_count = 0
    filelist = sorted(os.listdir(FLAGS.dataset_dir))
    for file in filelist:
        #height, width = get_image_size(data)

        # a mask
        a_mask_path = os.path.join(FLAGS.dataset_dir, file, gt_mask_list[0])
        b_mask_path = os.path.join(FLAGS.dataset_dir, file, gt_mask_list[1])

        a_mask_image = os.listdir(a_mask_path)[0]
        b_mask_image = os.listdir(b_mask_path)[0]

        a_mask = imread(os.path.join(a_mask_path, a_mask_image))
        b_mask = imread(os.path.join(b_mask_path, b_mask_image))

        equal_sum = np.sum(a_mask == b_mask)
        equal_percentage = (equal_sum*2) / (a_mask.size + b_mask.size)
        print(file, " >>>> equal percentage : ", equal_percentage)

        if equal_percentage == 1.0:
            equal_count += 1

    # summary
    print(FLAGS.ground_truth_prefix, ">> equal count >>", equal_count, "/", len(filelist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='../../tmp/nucleus_detection/stage1_train',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--ground_truth_prefix',
        default='gt_mask_labels,gt_mask',
        type=str,
        help="ground_truth data prefix")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


