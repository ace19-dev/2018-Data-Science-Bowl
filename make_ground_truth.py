from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys

from PIL import Image
from skimage.io import imread, imshow

from utils.image_utils import read_image

import cv2
import matplotlib.pyplot as plt

import scipy.ndimage as ndi

# def get_image_size(data):
#     image_path = os.path.join(FLAGS.dataset_dir, data, 'images')
#     image = os.listdir(image_path)
#     img = Image.open(os.path.join(image_path, image[0]))
#
#     return img.height, img.width

def get_contour(img):
    # '''
    img_contour = np.zeros_like(img).astype(np.uint8)
    # http://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 1)
    return img_contour
    # '''
    '''
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_erosion.html
    _img_contour = (img / 255.0).astype(np.uint8)
    erosion = ndi.morphology.binary_erosion(_img_contour, border_value=1).astype(_img_contour.dtype)
    img_contour = _img_contour - erosion
    img_contour = (255.0 * img_contour).astype(np.uint8)
    return img_contour
    '''


def morpho_op(BW):
    s = [[0,1,0],[1,1,1],[0,1,0]]#structuring element (diamond shaped)
    m_morfo = ndi.binary_opening(BW,structure=s,iterations=1)
    m_morfo = ndi.binary_closing(m_morfo,structure=s,iterations=1)
    M_filled = ndi.binary_fill_holes(m_morfo,structure=s)
    return M_filled

def main(_):

    filelist = sorted(os.listdir(FLAGS.dataset_dir))
    filecount = 1
    for data in filelist:
        image_path = os.path.join(FLAGS.dataset_dir, data, 'images', data + '.png')
        img_shape = read_image(image_path).shape

        mask_path = os.path.join(FLAGS.dataset_dir, data, 'masks')
        mask_images = sorted(os.listdir(mask_path))
        mask = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.bool)
        if FLAGS.use_contour:
            contour = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.bool)
        for mask_file in mask_images:
            _mask = imread(os.path.join(mask_path, mask_file))
            #
            # fill the holes that remained
            _mask_f = morpho_op(_mask)
            # Rescale to 0-255 and convert to uint8
            _mask_f = (255.0 * _mask_f).astype(np.uint8)
            _mask |= _mask_f
            # imshow(np.squeeze(_mask2))
            # plt.show()
            #
            _mask = np.expand_dims(_mask, axis=-1)
            mask = np.maximum(mask, _mask)
            #
            if FLAGS.use_contour:
                _contour = get_contour(_mask)
                contour = np.maximum(contour, _contour)
                # imshow(np.squeeze(_contour))
                # plt.show()

        gt_path = os.path.join(FLAGS.ground_truth_dir, data, FLAGS.ground_truth_folder)
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)

        print(">> (%d / %d) %s" % (filecount, filelist.__len__(), data))
        filecount += 1

        # imshow(np.squeeze(contour))
        # plt.show()

        contour_of_mask = get_contour(mask)
        # imshow(np.squeeze(contour_of_mask))
        # plt.show()

        contour_final = contour - contour_of_mask
        # imshow(np.squeeze(contour_final))
        # plt.show()

        #imshow(np.squeeze(mask))
        #plt.show()

        #mask2 = mask - contour
        #imshow(np.squeeze(mask2))
        #plt.show()

        if FLAGS.use_contour:
            mask = mask - contour_final

        mask = np.squeeze(mask)
        # imshow(mask)
        # plt.show()
        img = Image.fromarray(mask)
        img.save(os.path.join(gt_path, data + '.png'))
        # img.show(title=X)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='../../dl_data/nucleus/stage1_train_valid_elas',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--ground_truth_dir',
        default='../../dl_data/nucleus/stage1_train_valid_elas',
        type=str,
        help="ground_truth data directory")

    parser.add_argument(
        '--ground_truth_folder',
        default='gt_mask',
        type=str,
        help="ground_truth folder")

    parser.add_argument(
        '--use_contour',
        default=True,
        type=bool,
        help="use contour")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


