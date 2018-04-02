"""
Simple U-Net implementation in TensorFlow

Objective: detect vehicles

y = f(X)

X: image (640, 960, 3)
y: mask (640, 960, 1)
   - binary image
   - background is masked 0
   - vehicle is masked 255

Loss function: maximize IOU

    (intersection of prediction & grount truth)
    -------------------------------
    (union of prediction & ground truth)

Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import argparse
import sys
import os
import datetime

import tensorflow as tf
import numpy as np

from six.moves import xrange

import matplotlib.pyplot as plt

from nets.unet import Unet
# from _dataset.dataset_loader import DataLoader

from input_data import Data
from input_data import DataLoader


IMG_WIDTH = 256
IMG_HEIGHT = 256


def _IOU(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + \
                  tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    loss = -_IOU(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    # other optimizer will be used
    # optim = tf.train.AdamOptimizer()
    optim = tf.train.MomentumOptimizer(0.0001, 0.99)
    return optim.minimize(loss, global_step=global_step)


def get_start_epoch_number(latest_check_point):
    chck = latest_check_point.split('-')
    chck.reverse()
    return int(chck[0]) + 1


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode") # training or not

    pred = Unet(X, mode, FLAGS)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted Mask", pred)
    tf.summary.image("Predicted Mask", pred)

    # Updates moving mean and moving variance for BatchNorm (train/inference)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    IOU_op = _IOU(pred, y)
    # IOU_op = -_IOU(pred, y)
    tf.summary.scalar("IOU", IOU_op)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.logdir + '/validation')

    saver = tf.train.Saver()

    start_epoch = 1
    epoch_from_ckpt = 0
    if FLAGS.ckpt_dir:
        saver.restore(sess, FLAGS.ckpt_dir)
        tmp = FLAGS.ckpt_dir
        tmp = tmp.split('-')
        tmp.reverse()
        epoch_from_ckpt = int(tmp[0])
        start_epoch = epoch_from_ckpt + 1

    if epoch_from_ckpt != FLAGS.epochs + 1:
        tf.logging.info('Training from epoch: %d ', start_epoch)

    # Saving as Protocol Buffer (pb)
    tf.train.write_graph(sess.graph_def,
                         FLAGS.train_dir,
                         'unet.pbtxt',
                         as_text=True)


    ############################
    # Get data
    ############################
    raw = Data(FLAGS.data_dir, FLAGS.validation_percentage)
    tr_data = DataLoader(raw.data_dir,
                         raw.get_data('training'),
                         FLAGS.batch_size)
    val_data = DataLoader(raw.data_dir,
                          raw.get_data('validation'),
                          FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(tr_data.dataset.output_types,
                                               tr_data.dataset.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    tr_init_op = iterator.make_initializer(tr_data.dataset)
    val_init_op = iterator.make_initializer(val_data.dataset)

    tr_batches_per_epoch = int(tr_data.data_size / FLAGS.batch_size)
    if tr_data.data_size % FLAGS.batch_size > 0:
        tr_batches_per_epoch += 1
    val_batches_per_epoch = int(val_data.data_size / FLAGS.batch_size)
    if val_data.data_size % FLAGS.batch_size > 0:
        val_batches_per_epoch += 1


    ############################
    # Training
    ############################
    print("{} Training start ++++++++ ".format(datetime.datetime.now()))
    for epoch in xrange(start_epoch, FLAGS.epochs + 1):
        tf.logging.info('epoch #%d start >>> ', epoch)

        sess.run(tr_init_op)
        for step in range(tr_batches_per_epoch):
            X_train, y_train = sess.run(next_batch)
            train_summary, accuracy, _ = \
                sess.run([summary_op, IOU_op, train_op],
                         feed_dict={X: X_train, y: y_train, mode: True})

            train_summary_writer.add_summary(train_summary, step)
            tf.logging.info('epoch #%d, step #%d/%d, accuracy(iou) %.5f%%' %
                            (epoch, step, tr_batches_per_epoch, accuracy))

        print("{} Validation start ++++++++ ".format(datetime.datetime.now()))
        total_val_accuracy = 0
        val_count = 0
        sess.run(val_init_op)
        for n in range(val_batches_per_epoch):
            X_val, y_val = sess.run(next_batch)
            val_summary, val_accuracy = \
                sess.run([summary_op, IOU_op],
                         feed_dict={X: X_val, y: y_val, mode: False})

            # total_val_accuracy += val_step_iou * X_val.shape[0]
            total_val_accuracy += val_accuracy
            val_count += 1

            val_summary_writer.add_summary(val_summary, epoch)
            tf.logging.info('step #%d/%d, accuracy(iou) %.5f%%' %
                            (n, val_batches_per_epoch, total_val_accuracy * 100))

        total_val_accuracy /= val_count
        tf.logging.info('step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (epoch, total_val_accuracy * 100, raw.get_size('validation')))

        checkpoint_path = os.path.join(FLAGS.train_dir, 'unet.ckpt')
        tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
        saver.save(sess, checkpoint_path, global_step=epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_test',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs')

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help="Batch size")

    parser.add_argument(
        '--logdir',
        type=str,
        default=os.getcwd() + '/models/retrain_logs',
        help="Tensorboard log directory")

    parser.add_argument(
        '--train_dir',
        type=str,
        default=os.getcwd() + '/models',
        help='Directory to write event logs and checkpoint.')

    parser.add_argument(
        '--reg',
        type=float,
        default=0.1,
        help="L2 Regularizer Term")

    parser.add_argument(
        '--ckpt_dir',
        type=str,
        # default=os.getcwd() + '/models/mobile.ckpt-50',
        default='',
        help="Checkpoint directory")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
