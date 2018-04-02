import os
import argparse
import sys
import datetime
import csv

import numpy as np
import tensorflow as tf

from nets.unet import Unet

from input_pred_data import Data
from input_pred_data import DataLoader

IMG_WIDTH = 256
IMG_HEIGHT = 256



def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# TODO
# def prob_to_rles(x, cutoff=0.5):
    # lab_img = label(x > cutoff)
    # for i in range(1, lab_img.max() + 1):
    #     yield rle_encoding(lab_img == i)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], name="X")
    mode = tf.placeholder(tf.bool, name="mode")  # training or not

    pred = Unet(X, mode, FLAGS)
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
    test_data = DataLoader(raw.data_dir,
                           raw.get_data('prediction'),
                           FLAGS.batch_size)

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
    start_time = datetime.datetime.now()
    print("Start test: {}".format(start_time))

    submission = dict()

    # Initialize iterator with the test dataset
    sess.run(test_init_op)
    for i in range(test_batches_per_epoch):
        batch_xs, fnames = sess.run(next_batch)
        pred = sess.run(pred,
                        feed_dict={
                            X: batch_xs,
                            mode: False,
                        })
        # TODO
        # size = len(fnames)
        # for n in xrange(0, size):
        #     submission[fnames[n].decode('UTF-8')] = id2name[pred[n]]
        #
        # count += size
        # print(count, ' completed')

    end_time = datetime.datetime.now()
    print('{} Data, End prediction: {}'.format(test_data.data_size, end_time))
    print('prediction waste time: {}'.format(end_time - start_time))


    # TODO
    # new_test_ids = []
    # rles = []
    # for n, id_ in enumerate(test_ids):
    #     rle = list(prob_to_rles(preds_test_upsampled[n]))
    #     rles.extend(rle)
    #     new_test_ids.extend([id_] * len(rle))
    #
    # # Create submission DataFrame
    # sub = pd.DataFrame()
    # sub['ImageId'] = new_test_ids
    # sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    # sub.to_csv('sub-dsbowl2018-1.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        default='/home/ace19/dl-data/nucleus_detection/stage1_test',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--batch_size',
        default=64,
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