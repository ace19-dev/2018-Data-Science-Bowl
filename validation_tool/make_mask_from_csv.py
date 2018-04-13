from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import shutil

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Color map


def get_image_size(imageId):
    height = 0
    width = 0

    try:
        image_path = os.path.join(FLAGS.dataset_dir, imageId, 'images')
        image = os.listdir(image_path)
        img = Image.open(os.path.join(image_path, image[0]))
        height = img.height
        width = img.width
    except:
        print("get_image_size exception")

    return height, width

def remove_eval_dir(imageId, target_path):
    dir_path = os.path.join(FLAGS.dataset_dir, imageId, target_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def save_to_image(imageId, data, target_path, prefix):
    mask_path = os.path.join(FLAGS.dataset_dir, imageId, target_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    target = os.path.join(mask_path, prefix + imageId + '.png')

    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    # save
    img = Image.fromarray(rescaled)
    img.save(target)
    print("save_to_image >>>", mask_path)


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
    per_image_index = 0
    for imageId in data_frame['ImageId']:
        #print(imageId)

        if preImageId != imageId:
            # make gt_mask for preImageId
            if image.size > 0:
                image_2 = image.reshape(width, height).T
                #plt.imshow(image_2, cm.gray)
                #plt.show()
                # save
                save_to_image(preImageId, image_2, FLAGS.gt_mask_dir, '')

            preImageId = imageId
            imageIdChanged = True
            per_image_index = 0
            remove_eval_dir(imageId, FLAGS.mask_dir)
            remove_eval_dir(imageId, FLAGS.gt_mask_dir)
            image = np.zeros(0)
        else:
            imageIdChanged = False

        height, width = get_image_size(imageId)
        #print(imageId, " >>>>> ", width, height)
        if height == 0 or width == 0:
            continue

        mask_info = data_frame['EncodedPixels'][index]
        mask_array = np.fromstring(mask_info, sep=" ", dtype=np.uint32)
        #print(mask_info)
        #print(mask_array)

        per_image = np.zeros((height, width), dtype=np.uint8).flatten()
        if imageIdChanged == True:
            image = np.zeros((height, width), dtype=np.uint8).flatten()
        for i in range(0, len(mask_array), 2):
            start_pos = mask_array[i]
            mask_length = mask_array[i + 1]
            #print("start >> ", start_pos)
            #print("mask_length >> ", mask_length)
            for j in range(mask_length):
                per_image[start_pos - 1 + j] = 1
                image[start_pos - 1 + j] = 1

        # show per each
        per_image_2 = per_image.reshape(width, height).T
        #plt.imshow(per_image, cm.gray)
        #plt.show()

        # save
        save_to_image(preImageId, per_image_2, FLAGS.mask_dir, str(per_image_index))
        per_image_index += 1

        index += 1
        #break

    if image.size > 0:
        image_2 = image.reshape(width, height).T
        #plt.imshow(image_2, cm.gray)
        #plt.show()
        # save
        save_to_image(preImageId, image_2, FLAGS.gt_mask_dir, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labels_path',
        default='../result_2/submission-nucleus_det_stage2-100.csv',
        # default='../../../dl_data/nucleus/stage1_train_labels/stage1_train_labels.csv',
        # default='../../../dl_data/nucleus/stage1_solution/stage1_solution.csv',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--dataset_dir',
        # default='../../../dl_data/nucleus/stage1_test',
        default='../../../dl_data/nucleus/stage2_test_final',
        # default='../../../dl_data/nucleus/stage1_train',
        type=str,
        help="Labels directory")

    parser.add_argument(
        '--mask_dir',
        default='eval_mask',
        # default='solution_eval_mask',
        type=str,
        help="mask directory name")

    parser.add_argument(
        '--gt_mask_dir',
        default='eval_gt_mask',
        # default='solution_eval_gt_mask',
        type=str,
        help="gt_mask directory name")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)