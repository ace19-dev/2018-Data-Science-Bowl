
from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import random
import sys

from skimage.io import imread, imshow
from skimage.transform import resize

from PIL import Image
import matplotlib.pyplot as plt


# TFRecords convertion parameters.
RANDOM_SEED = 777
SAMPLES_PER_FILES = 200


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


# Basic model parameters as external flags.
FLAGS = None


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _process_image(dataset_dir, filename):
    """Process a image and annotation file.

    Args:
      dataset_dir: string, path to an image dir/image name.
      filename: Image dir/name to add to the TFRecord.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    img_path = os.path.join(dataset_dir, filename, 'images', filename) + '.png'
    img_bytes = tf.gfile.FastGFile(img_path, 'rb').read()
    # image = imread(img_path)[:, :, :IMG_CHANNELS]
    # plt.imshow(image)
    # plt.title(filename)
    # plt.show()

    mask_path = os.path.join(dataset_dir, filename, 'masks')
    mask_images = sorted(os.listdir(mask_path))
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in mask_images:
        _mask = imread(os.path.join(mask_path, mask_file))
        _mask = np.expand_dims(
            resize(_mask,
                   (IMG_HEIGHT, IMG_WIDTH),
                   mode='constant',
                   preserve_range=True),
            axis=-1)
        mask = np.maximum(mask, _mask)

    # imshow(np.squeeze(mask))
    # plt.show()

    # mask = np.squeeze(mask)
    # img = Image.fromarray(mask)
    # img.save(os.path.join(mask_path, '_mask.png'))
    # img.show()

    return img_bytes, np.squeeze(mask).tobytes()


def _convert_to_example(image, label):
    """Build an Example proto for an image example.

    Args:
      image: string, PNG encoding of RGB image;
      label: list of mask, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(
        feature={
            # 'image/height': int64_feature(shape[0]),
            # 'image/width': int64_feature(shape[1]),
            # 'image/channels': int64_feature(shape[2]),
            # 'image/shape': int64_feature(shape),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image),
            'masks/encoded': bytes_feature(label)
        }
    ))

    return example


def _add_to_tfrecord(dataset_dir, filename, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      filename: Image dir/name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image, label = _process_image(dataset_dir, filename)
    example = _convert_to_example(image, label)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def main(_):

    if not tf.gfile.Exists(FLAGS.dataset_dir):
        raise ValueError('You must supply the _dataset directory with --dataset_dir')

    # Dataset filenames, and shuffling.
    # path = os.path.join(FLAGS.dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(FLAGS.dataset_dir))
    if FLAGS.shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process _dataset files.
    i = 0
    idx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(FLAGS.output_dir, FLAGS.dataset_type, idx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                # img_name = filename[:-4]
                _add_to_tfrecord(FLAGS.dataset_dir, filename, tfrecord_writer)
                i += 1
                j += 1
            idx += 1

    print('\nFinished converting the nucleus detection _dataset!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_test',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--output_dir',
        default='/home/ace19/dl-data/nucleus_detection/stage1_train_tfrecord',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_test_tfrecord',
        type=str,
        help="Output data directory")

    # parser.add_argument(
    #     '--num_threads',
    #     type=int,
    #     default=1,
    #     help="Number of threads")
    #
    # parser.add_argument(
    #     '--num_shards',
    #     type=int,
    #     default=1,
    #     help="Number of shards in training TFRecord files")

    parser.add_argument(
        '--shuffling',
        type=bool,
        default=True,
        help="Shuffle or not")

    parser.add_argument(
        '--dataset_type',
        type=str,
        default='train',
        help="train or test")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
