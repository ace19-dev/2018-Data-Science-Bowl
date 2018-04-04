""" Collection of methods to compute the score.

1. We start with a true and predicted mask, corresponding to one train image.

2. The true mask is segmented into different objects. Here lies a main source
of error. Overlapping or touching nuclei are not separated but are labeled as
one object. This means that the target mask can contain less objects than
those that have been originally identified by humans.

3. In the same manner the predicted mask is segmented into different objects.

4. We compute all intersections between the objects of the true and predicted
masks. Starting with the largest intersection area we assign true objects to
predicted ones, until there are no true/pred objects left that overlap.
We then compute for each true/pred object pair their corresponding intersection
over union (iou) ratio.

5. Given some threshold t we count the object pairs that have an iou > t, which
yields the number of true positives: tp(t). True objects that have no partner are
counted as false positives: fp(t). Likewise, predicted objects without a counterpart
a counted as false negatives: fn(t).

6. Now, we compute the precision tp(t)/(tp(t)+fp(t)+fn(t)) for t=0.5,0.55,0.60,...,0.95
and take the mean value as the final precision (score).
"""


import os
import argparse
import sys
import datetime
import csv

from six.moves import xrange
from skimage.transform import resize
from skimage.morphology import label
# from scipy.ndimage.measurements import label
import pandas as pd

import numpy as np
import tensorflow as tf

from nets.unet import Unet, Unet2
from utils.data_oper import read_test_data_properties, mask_to_rle, \
                                trsf_proba_to_binary, rle_to_mask
from input_pred_data import Data
from input_pred_data import DataLoader

IMG_WIDTH = 256
IMG_HEIGHT = 256


# def rle_encoding(x):
#     dots = np.where(x.T.flatten() == 1)[0]
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b > prev + 1): run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths
#
#
# def prob_to_rles(x, cutoff=0.5):
#     lab_img = label(x > cutoff)
#     for i in range(1, lab_img.max() + 1):
#         yield rle_encoding(lab_img == i)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], name="X")
    mode = tf.placeholder(tf.bool, name="mode")  # training or not
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    # pred = Unet(X, mode, FLAGS)
    pred = Unet2(X, dropout_prob, FLAGS)
    # evaluation = tf.argmax(logits, 1)

    sess.run(tf.global_variables_initializer())

    # Restore variables from training checkpoints.
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' % (
            ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return


    ############################
    # Get data
    ############################
    raw = Data(FLAGS.data_dir)
    test_data = DataLoader(raw.get_data(), FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(test_data.dataset.output_types,
                                               test_data.dataset.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    test_init_op = iterator.make_initializer(test_data.dataset)

    test_batches_per_epoch = int(test_data.data_size / FLAGS.batch_size)
    if test_data.data_size % FLAGS.batch_size > 0:
        test_batches_per_epoch += 1


    ##################################################
    # start test & make csv file.
    ##################################################

    # Read basic properties of test images.
    test_df = read_test_data_properties(FLAGS.data_dir, 'images')

    test_pred_proba = []
    test_pred_fnames = []

    start_time = datetime.datetime.now()
    print("start test: {}".format(start_time))

    # Initialize iterator with the test dataset
    sess.run(test_init_op)
    for i in range(test_batches_per_epoch):
        batch_xs, fnames = sess.run(next_batch)
        prediction = sess.run(pred,
                              feed_dict={
                                  X: batch_xs,
                                  # mode: False,
                                  dropout_prob: 1.0
                              })

        test_pred_proba.extend(prediction)
        test_pred_fnames.extend(fnames)

    end_time = datetime.datetime.now()
    print('end test: {}'.format(test_data.data_size, end_time))
    print('test waste time: {}'.format(end_time - start_time))

    # Transform propabilities into binary values 0 or 1.
    test_pred = trsf_proba_to_binary(test_pred_proba)

    # Resize predicted masks to original image size.
    original_size_test_pred = []
    for i in range(len(test_pred)):
        res_mask = trsf_proba_to_binary(
            resize(
                np.squeeze(test_pred[i]),
                (test_df.loc[i, 'img_height'], test_df.loc[i, 'img_width']),
                mode='constant',
                preserve_range=True)
        )
        original_size_test_pred.append(res_mask)
    original_size_test_pred = np.array(original_size_test_pred)

    # # Inspect a test prediction and check run length encoding.
    # for n, id_ in enumerate(test_df['img_id']):
    #     fname = test_pred_fnames[n]
    #     mask = original_size_test_pred[n]
    #     rle = list(mask_to_rle(mask))
    #     mask_rec = rle_to_mask(rle, mask.shape)
    #     print('no:{}, {} -> Run length encoding: {} matches, {} misses'.format(
    #         n, fname, (mask_rec == mask).sum(), (mask_rec != mask).sum()))

    # Run length encoding of predicted test masks.
    test_pred_rle = []
    test_pred_ids = []
    for n, _id in enumerate(test_df['img_id']):
        min_object_size = 20 * test_df.loc[n, 'img_height'] * test_df.loc[n, 'img_width'] / (256 * 256)
        rle = list(mask_to_rle(original_size_test_pred[n], min_object_size=min_object_size))
        test_pred_rle.extend(rle)
        test_pred_ids.extend([_id] * len(rle))

    # Create submission DataFrame
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    sub = pd.DataFrame()
    sub['ImageId'] = test_pred_ids
    sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(os.path.join(FLAGS.result_dir, 'submission-nucleus_det-' + global_step + '.csv'),
               index=False)
    sub.head()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        default='/home/acemc19/dl-data/nucleus_detection/stage1_test',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help="Batch size")

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.getcwd() + '/models',
        help='Directory to write event logs and checkpoint.')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.getcwd() + '/result',
        help='Directory to write submission.csv file.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)