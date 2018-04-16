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

from nets.unet import Unet_32_512, Unet_64_1024
# from _dataset.dataset_loader import DataLoader

from input_data import Data
from input_data import DataLoader

FLAGS = None

from utils.checkmate import BestCheckpointSaver


def IOU(y_pred, y_true):
    """Returns a (approx) batch_norm_wrapper score

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

    # smooth = 1.
    # intersection = 2. * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    # denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    # return tf.reduce_mean(intersection / denominator)


def get_start_epoch_number(latest_check_point):
    chck = latest_check_point.split('-')
    chck.reverse()
    return int(chck[0]) + 1


def main(_):
    # specify GPU
    if FLAGS.gpu_index:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, 3], name="X")
    GT = tf.placeholder(tf.float32, shape=[None, FLAGS.label_size, FLAGS.label_size, 1], name="GT")
    mode = tf.placeholder(tf.bool, name="mode") # training or not

    if FLAGS.use_64_channel:
        pred = Unet_64_1024(X, mode, FLAGS)
    else:
        pred = Unet_32_512(X, mode, FLAGS)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted Mask", pred)
    tf.summary.image("Predicted Mask", pred)

    # IOU is
    #
    # (the area of intersection)
    # --------------------------
    # (the area of two boxes)
    iou_op = IOU(pred, GT)

    loss = -iou_op
    tf.summary.scalar("loss", loss)

    # Updates moving mean and moving variance for BatchNorm (train/inference)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # other optimizer will be used
        train_op = tf.train.MomentumOptimizer(0.001, 0.99).minimize(loss)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.logdir + '/validation')

    saver = tf.train.Saver()

    # For, checkpoint saver
    if FLAGS.best_train_dir:
        best_ckpt_saver = BestCheckpointSaver(title='unet.ckpt', save_dir=FLAGS.best_train_dir, num_to_keep=3, maximize=True)

    start_epoch = 1
    epoch_from_ckpt = 0
    if FLAGS.ckpt_path:
        saver.restore(sess, FLAGS.ckpt_path)
        tmp = FLAGS.ckpt_path
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
                         FLAGS.img_size,
                         FLAGS.label_size,
                         FLAGS.batch_size)
    val_data = DataLoader(raw.data_dir,
                          raw.get_data('validation'),
                          FLAGS.img_size,
                          FLAGS.label_size,
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
    print("{} Training start ... ".format(datetime.datetime.now()))
    for epoch in xrange(start_epoch, FLAGS.epochs + 1):
        print('{} Training epoch-{} start >> '.format(datetime.datetime.now(), epoch))

        sess.run(tr_init_op)
        for step in range(tr_batches_per_epoch):
            X_train, y_train = sess.run(next_batch)
            train_summary, accuracy, _, _ = \
                sess.run([summary_op, iou_op, train_op, increment_global_step],
                         feed_dict={X: X_train,
                                    GT: y_train,
                                    mode: True}
                         )

            train_summary_writer.add_summary(train_summary, (epoch-start_epoch)*tr_batches_per_epoch+step)
            tf.logging.info('epoch #%d, step #%d/%d, accuracy(iou) %.5f%%' %
                            (epoch, step, tr_batches_per_epoch, accuracy))

        print("{} Validation start ... ".format(datetime.datetime.now()))
        total_val_accuracy = 0
        val_count = 0
        sess.run(val_init_op)
        for n in range(val_batches_per_epoch):
            X_val, y_val = sess.run(next_batch)
            val_summary, val_accuracy = \
                sess.run([summary_op, iou_op],
                         feed_dict={X: X_val,
                                    GT: y_val,
                                    mode: False}
                         )

            # total_val_accuracy += val_step_iou * X_val.shape[0]
            total_val_accuracy += val_accuracy
            val_count += 1

            val_summary_writer.add_summary(val_summary, (epoch-start_epoch)*val_batches_per_epoch+n)
            tf.logging.info('step #%d/%d, accuracy(iou) %.5f%%' %
                            (n, val_batches_per_epoch, val_accuracy * 100))

        total_val_accuracy /= val_count
        tf.logging.info('step %d: Validation accuracy = %.2f%% (N=%d)' %
                        (epoch, total_val_accuracy * 100, raw.get_size('validation')))

        # save checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, 'unet.ckpt')
        tf.logging.info('Saving to "%s-%d"', checkpoint_path, epoch)
        saver.save(sess, checkpoint_path, global_step=epoch)

        # save best checkpoint
        if FLAGS.best_train_dir:
            best_ckpt_saver.handle(total_val_accuracy, sess, global_step, epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        # default='../../dl_data/nucleus/stage1_train_valid_elas',
        default='../../dl_data/nucleus/stage1_train',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')

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
        '--best_train_dir',
        type=str,
        # default=os.getcwd() + '/models_best',
        default=None,
        help="Directory to write best checkpoint.")

    # parser.add_argument(
    #     '--reg',
    #     type=float,
    #     default=0.1,
    #     help="L2 Regularizer Term")

    parser.add_argument(
        '--ckpt_path',
        type=str,
        # default=os.getcwd() + '/models/unet.ckpt-20',
        default='',
        help="Checkpoint directory")

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs')

    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help="Batch size")

    parser.add_argument(
        '--img_size',
        type=int,
        # default=256,
        default=512,
        # default=572,
        help="Image height and width")

    parser.add_argument(
        '--label_size',
        type=int,
        # default=256,
        default=512,
        # default=388,
        help="Label height and width")

    parser.add_argument(
        '--conv_padding',
        type=str,
        default='same',
        # default='valid',
        help="conv padding. if your img_size is 572 and, conv_padding is valid then the label_size is 388")

    parser.add_argument(
        '--use_64_channel',
        type=bool,
        default=True,
        # default=False,
        help="If you set True then use the Unet_64_1024. otherwise use the Unet_32_512")

    parser.add_argument(
        '--gpu_index',
        type=str,
        # default=None,
        default='0',
        # default='1',
        help="Set the gpu index. If you not sepcify then auto")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
