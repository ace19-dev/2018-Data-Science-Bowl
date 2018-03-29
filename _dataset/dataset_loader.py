from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob

import tensorflow as tf


IMG_WIDTH = 256
IMG_HEIGHT = 256


def absoluteFilePaths(directory):
    file_paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            file_paths.append(os.path.abspath(os.path.join(dirpath, f)))

    return file_paths


class DataLoader(object):

  def __init__(self, tfrecord_dir, batch_size, shuffle=True):
    # create _dataset, Creating a source
    filenames = absoluteFilePaths(tfrecord_dir)
    dataset = tf.data.TFRecordDataset(filenames)

    # shuffle the first `buffer_size` elements of the _dataset
    #  Make sure to call tf.data.Dataset.shuffle() before applying the heavy transformations
    # (like reading the images, processing them, batching...).
    if shuffle:
      dataset = dataset.shuffle(buffer_size= 100 * batch_size)

    # distinguish between train/infer. when calling the parsing functions
    # transform to images, preprocess, repeat, batch...
    dataset = dataset.map(self._parse_function, num_parallel_calls=8)

    dataset = dataset.prefetch(buffer_size = 10 * batch_size)

    # create a new _dataset with batches of images
    dataset = dataset.batch(batch_size)

    self.dataset = dataset


  def _parse_function(self, example_proto):
    features = {'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'masks/encoded': tf.FixedLenFeature((), tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)

    image_decoded = tf.image.decode_png(parsed_features["image/encoded"], channels=3)
    image_resized = tf.image.resize_images(image_decoded, [IMG_HEIGHT, IMG_WIDTH])

    mask_decoded = tf.image.decode_png(parsed_features["masks/encoded"], channels=1)
    mask_resized = tf.image.resize_images(mask_decoded, [IMG_HEIGHT, IMG_WIDTH])

    return image_resized, mask_resized


  # def _parse_function(self, filename, label):
  #   image_string = tf.read_file(filename)
  #   image_decoded = tf.image.decode_png(image_string, channels=3)
  #   # image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  #   # image = tf.cast(image_decoded, tf.float32)
  #   image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  #   # Finally, rescale to [-1,1] instead of [0, 1)
  #   # image = tf.subtract(image, 0.5)
  #   # image = tf.multiply(image, 2.0)
  #   return image, label