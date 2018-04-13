import os
import sys
import argparse
import tqdm
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf

# For using image generation
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from utils.image_utils import read_image

FLAGS = None

RANDOM_SEED = 54989


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


def make_aug_dir(prefix_name):
    randomString = str(uuid.uuid4()).replace("-", "")
    _new = FLAGS.aug_prefix + prefix_name + randomString

    return _new


def generate_images(image_generator, src_path, target_dir, seed=None):
    """Generate new images."""
    img = load_img(src_path, interpolation='nearest')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Generate new set of images
    batches = 1
    for batch in image_generator.flow(x,
                                      batch_size=1,
                                      shuffle=False,
                                      seed=seed,
                                      save_to_dir=target_dir):

        batches += 1
        if batches > 1:
            break  # otherwise the generator would loop indefinitely


def main(_):
    img_gen = ImageDataGenerator(# rotation_range=90.,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 brightness_range=[1.0, 1.2],
                                 fill_mode='reflect',
                                 horizontal_flip=True,
                                 vertical_flip=True)

    train_info = read_train_data_properties(FLAGS.train_dir)

    # image_augmentation
    for i, filename in tqdm.tqdm(enumerate(train_info['image_path']), total=len(train_info)):
        _name = os.path.basename(filename)
        for n in range(FLAGS.aug_count):
            seed = np.random.randint(RANDOM_SEED)
            data_path = os.path.join(FLAGS.train_dir, make_aug_dir(_name[:10]))

            target_img_dir = os.path.join(data_path, 'images')
            target_mask_dir = os.path.join(data_path, 'gt_mask')

            generate_images(img_gen, train_info['image_path'].loc[i], target_img_dir, seed=seed)
            generate_images(img_gen, train_info['mask_path'].loc[i], target_mask_dir, seed=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default='../../../dl_data/nucleus/stage1_train',
        type=str,
        help="Train Data directory")

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