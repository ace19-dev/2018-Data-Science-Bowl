from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Color map


def get_image_size(imageId):
    image_path = os.path.join(FLAGS.dataset_dir, imageId, 'images')
    image = os.listdir(image_path)
    img = Image.open(os.path.join(image_path, image[0]))

    return img.height, img.width


def save_to_image(imageId, data):
    gt_path = os.path.join(FLAGS.dataset_dir, imageId, FLAGS.ground_truth_prefix)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    target = os.path.join(gt_path, imageId + '.png')

    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    # save
    img = Image.fromarray(rescaled)
    img.save(target)
    print("save_to_image >>> ", imageId)


def main(_):
    # open csv
    csv_filename = os.path.join(FLAGS.labels_path)
    data_frame = pd.read_csv(csv_filename)
    # print(data_frame['ImageId'])
    # print(data_frame['EncodedPixels'])

    # making gt_mask for ImageId
    index = 0
    preImageId = ''
    imageIdChanged = False
    image = np.zeros(0)
    height = 0
    width = 0
    for imageId in data_frame['ImageId']:
        #print(imageId)

        if preImageId != imageId:
            # make gt_mask for preImageId
            if image.size > 0:
                image_2 = image.reshape(width, height).T
                #plt.imshow(image_2, cm.gray)
                #plt.show()
                # save
                save_to_image(preImageId, image_2)

            preImageId = imageId
            imageIdChanged = True
        else:
            imageIdChanged = False

        height, width = get_image_size(imageId)
        #print(imageId, " >>>>> ", width, height)

        mask_info = data_frame['EncodedPixels'][index]
        mask_array = np.fromstring(mask_info, sep=" ", dtype=np.uint32)
        #print(mask_info)
        #print(mask_array)

        if imageIdChanged == True:
            image = np.zeros((height, width), dtype=np.uint8).flatten()
        for i in range(0, len(mask_array), 2):
            start_pos = mask_array[i]
            mask_length = mask_array[i + 1]
            #print("start >> ", start_pos)
            #print("mask_length >> ", mask_length)
            for j in range(mask_length):
                image[start_pos - 1 + j] = 1

        # show per each
        #image_2 = image.reshape(width, height).T
        #plt.imshow(image_2, cm.gray)
        #plt.show()

        index += 1
        #break

    if image.size > 0:
        image_2 = image.reshape(width, height).T
        #plt.imshow(image_2, cm.gray)
        #plt.show()
        # save
        save_to_image(preImageId, image_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labels_path',
        default='../../tmp/nucleus_detection/stage1_train_labels/stage1_train_labels.csv',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--dataset_dir',
        default='../../tmp/nucleus_detection/stage1_train',
        type=str,
        help="Labels directory")

    parser.add_argument(
        '--ground_truth_prefix',
        default='gt_mask_labels',
        type=str,
        help="ground_truth data prefix")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)