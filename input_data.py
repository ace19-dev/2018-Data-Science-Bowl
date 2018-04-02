from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os.path
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
RANDOM_SEED = 888

HEIGHT = 256
WIDTH = 256


def which_set(filename, validation_percentage):
  """Determines which data partition the file should belong to.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
  Returns:
    String, one of 'training', 'validation'.
  """
  base_name = os.path.basename(filename)
  hash_name_hashed = hashlib.sha1(compat.as_bytes(base_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  else:
    result = 'training'
  return result


class Data(object):
  def __init__(self, data_dir, validation_percentage):
    self.data_dir = data_dir
    self._prepare_data_index(validation_percentage)


  def get_data(self, mode):
    return self.data_index[mode]


  def get_size(self, mode):
    """Calculates the number of samples in the _dataset partition.
    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.
    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])


  def _prepare_data_index(self, validation_percentage):
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    self.data_index = {'validation': [], 'training': []}
    data_path = os.listdir(self.data_dir)
    for image_path in data_path:
      set_index = which_set(image_path, validation_percentage)
      self.data_index[set_index].append({'image': image_path})

    # Make sure the ordering is random.
    for set_index in ['validation', 'training']:
      random.shuffle(self.data_index[set_index])


class DataLoader(object):

  def __init__(self, data_dir, data, batch_size, shuffle=True):

    self.data_size = len(data)

    images, labels = self._get_data(data_dir, data)

    # create _dataset, Creating a source
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

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


  def _get_data(self, data_dir, data):
    image_paths = np.array(data)
    mask_paths = np.array(data)

    for idx, image_path in enumerate(image_paths):
        image_paths[idx] = \
            os.path.join(data_dir, image_path['image'], 'images', image_path['image']) + '.png'
        mask_paths[idx] = \
          os.path.join(data_dir, image_path['image'], 'gt_mask', image_path['image']) + '.png'

    # convert lists to TF tensor
    image_paths = convert_to_tensor(image_paths, dtype=dtypes.string)
    mask_paths = convert_to_tensor(mask_paths, dtype=dtypes.string)

    return image_paths, mask_paths


  def _parse_function(self, image_file, label_file):
    image_string = tf.read_file(image_file)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, HEIGHT, WIDTH)
    # image = tf.cast(image_resized, tf.float32)
    image = tf.image.convert_image_dtype(image_resized, dtype=tf.float32)
    # Finally, rescale to [-1,1] instead of [0, 1)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)

    label_string = tf.read_file(label_file)
    label_decoded = tf.image.decode_png(label_string, channels=1)
    label_resized = tf.image.resize_image_with_crop_or_pad(label_decoded, HEIGHT, WIDTH)
    # label = tf.cast(label_resized, tf.float32)
    label = tf.image.convert_image_dtype(label_resized, dtype=tf.float32)
    # Finally, rescale to [-1,1] instead of [0, 1)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)

    return image, label