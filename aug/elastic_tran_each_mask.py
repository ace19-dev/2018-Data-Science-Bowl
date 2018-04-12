import os
import sys
import argparse
import tqdm
import cv2
import uuid

import numpy as np
import tensorflow as tf

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

FLAGS = None

def main(_):
    train_ids = next(os.walk(FLAGS.train_dir))[1]
    train_ids.sort()

    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    for aug_count in range(0, FLAGS.aug_count):
        print('elastic transformation step {}'.format(aug_count + 1))
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = FLAGS.train_dir + id_
            image_ = cv2.imread(path + '/images/' + id_ + '.png')
            print('image name : ', path + '/images/' + id_ + '.png')

            # Read Masks
            flag = False
            maks_list = next(os.walk(path + '/masks/'))[2]
            print('maks size : ', len(maks_list))

            for mask_file in maks_list:

                mask_ = cv2.imread(path + '/masks/' + mask_file)
                if flag:
                    image = np.concatenate((image, mask_), axis=2)
                else:
                    image = np.concatenate((image_, mask_), axis=2)
                    flag = True

            # image = np.concatenate((image_, mask_), axis=2)

            alpha = image.shape[1] * 2
            sigma = image.shape[1] * 0.08
            alpha_affine = image.shape[1] * 0.05
            random_state = np.random.RandomState(None)

            shape = image.shape
            shape_size = shape[:2]

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32(
                [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                 center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            try:
                image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
            except:
                print('exception')
                continue

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

            im_merge_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

            randomString = str(uuid.uuid4()).replace("-", "")
            new_id = id_[:10] + FLAGS.aug_prefix + randomString
            os.mkdir(FLAGS.train_dir + new_id)
            os.mkdir(FLAGS.train_dir + new_id + '/images/')
            im_t = im_merge_t[..., 0:3]
            cv2.imwrite(FLAGS.train_dir + new_id + '/images/' + new_id + '.png', im_t)

            index = 3
            os.mkdir(FLAGS.train_dir + new_id + '/masks/')
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_t = im_merge_t[..., index:index+3]
                mask_t = cv2.cvtColor(mask_t, cv2.COLOR_RGB2GRAY)
                index = index + 3
                cv2.imwrite(FLAGS.train_dir + new_id + '/masks/' + mask_file, mask_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default='../../../dl_data/nucleus/stage1_train_valid/',
        type=str,
        help="Train Data directory")

    parser.add_argument(
        '--aug_prefix',
        default='_elastic_',
        type=str,
        help="prefix name of augmentation")

    parser.add_argument(
        '--aug_count',
        type=int,
        default=3,
        help="Count of augmentation")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)