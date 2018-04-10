from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys

from PIL import Image
from skimage.io import imread, imshow

from utils.image_utils import read_image

import tqdm
import glob
import cv2
import matplotlib.pyplot as plt

# def get_image_size(data):
#     image_path = os.path.join(FLAGS.dataset_dir, data, 'images')
#     image = os.listdir(image_path)
#     img = Image.open(os.path.join(image_path, image[0]))
#
#     return img.height, img.width

def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 1)
    return img_contour

def overlay_contours(images_dir, subdir_name, target_dir, touching_only=False):
    train_dir = os.path.join(images_dir, subdir_name)
    contours = []
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_contour(image))
        if touching_only:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128. + 255., 255., 0.).astype(np.uint8)
        else:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        #target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        #os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        #imwrite(target_filepath, overlayed_masks)
        contours.append(overlayed_masks)

    return contours

def main(_):

    filelist = sorted(os.listdir(FLAGS.dataset_dir))
    for data in filelist:
        image_path = os.path.join(FLAGS.dataset_dir, data, 'images', data + '.png')
        img_shape = read_image(image_path).shape

        mask_path = os.path.join(FLAGS.dataset_dir, data, 'masks')
        mask_images = sorted(os.listdir(mask_path))
        mask = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.bool)
        if FLAGS.use_countour:
            countour = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.bool)
        for mask_file in mask_images:
            _mask = imread(os.path.join(mask_path, mask_file))
            _mask = np.expand_dims(_mask, axis=-1)
            mask = np.maximum(mask, _mask)
            #
            if FLAGS.use_countour:
                _countour = get_contour(_mask)
                countour = np.maximum(countour, _countour)
                #imshow(np.squeeze(_countour))
                #plt.show()

        gt_path = os.path.join(FLAGS.ground_truth_dir, data, FLAGS.ground_truth_folder)
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)

        print(data)

        #imshow(np.squeeze(countour))
        #plt.show()

        #imshow(np.squeeze(mask))
        #plt.show()

        if FLAGS.use_countour:
            mask = mask - countour
        #imshow(np.squeeze(mask))
        #plt.show()

        mask = np.squeeze(mask)
        img = Image.fromarray(mask)
        img.save(os.path.join(gt_path, data + '.png'))
        # img.show(title=X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='../../dl_data/nucleus/stage1_train',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--ground_truth_dir',
        default='../../dl_data/nucleus/stage1_train',
        type=str,
        help="ground_truth data directory")

    parser.add_argument(
        '--ground_truth_folder',
        default='gt_mask2',
        type=str,
        help="ground_truth folder")

    parser.add_argument(
        '--use_countour',
        default=True,
        type=bool,
        help="use countour")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


