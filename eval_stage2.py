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

from nets.unet import Unet_32_512, Unet_64_1024
from utils.oper_utils2 import read_test_data_properties, mask_to_rle, \
                                trsf_proba_to_binary, rle_to_mask
from input_pred_data import Data
from input_pred_data import DataLoader

from PIL import Image
import scipy.ndimage as ndi

FLAGS = None

def get_image_size(imageId):
    image_path = os.path.join(FLAGS.data_dir, imageId, 'images')
    image = os.listdir(image_path)
    img = Image.open(os.path.join(image_path, image[0]))

    return img.height, img.width

def morpho_op(BW):
    s = [[0,1,0],[1,1,1],[0,1,0]]#structuring element (diamond shaped)
    m_morfo = ndi.binary_opening(BW,structure=s,iterations=1)
    m_morfo = ndi.binary_closing(m_morfo,structure=s,iterations=1)
    M_filled = ndi.binary_fill_holes(m_morfo,structure=s)
    return M_filled

def main(_):
    # specify GPU
    if FLAGS.gpu_index:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    tf.logging.set_verbosity(tf.logging.INFO)

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.InteractiveSession(config=config)

    X = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, 3], name="X")
    mode = tf.placeholder(tf.bool, name="mode")  # training or not

    if FLAGS.use_64_channel:
        pred = Unet_64_1024(X, mode, FLAGS)
    else:
        pred = Unet_32_512(X, mode, FLAGS)
    # evaluation = tf.argmax(logits, 1)

    sess.run(tf.global_variables_initializer())

    # Restore variables from training checkpoints.
    saver = tf.train.Saver()
    checkpoint_path = None
    if FLAGS.checkpoint_dir and FLAGS.checkpoint_file:
        checkpoint_path = FLAGS.checkpoint_dir+'/'+FLAGS.checkpoint_file
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_path = ckpt.model_checkpoint_path

    if checkpoint_path:
        saver.restore(sess, checkpoint_path)
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' % (
            checkpoint_path, global_step))
    else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return


    ############################
    # Get data
    ############################
    raw = Data(FLAGS.data_dir)
    test_data = DataLoader(raw.get_data(), FLAGS.img_size, FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(test_data.dataset.output_types,
                                               test_data.dataset.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    test_init_op = iterator.make_initializer(test_data.dataset)

    test_batches_per_epoch = int(test_data.data_size / FLAGS.batch_size)
    if test_data.data_size % FLAGS.batch_size > 0:
        test_batches_per_epoch += 1


    ##################################################
    # prepare
    ##################################################

    # Create result_dir
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    # Delete existed submission file
    filename = os.path.join(FLAGS.result_dir, 'submission-nucleus_det_stage2-' + global_step + '.csv')
    if os.path.exists(filename):
        os.remove(filename)


    ##################################################
    # start test & make csv file.
    ##################################################

    start_time = datetime.datetime.now()
    print("start test: {}".format(start_time))

    total_process_count = test_data.data_size
    process_count = 0

    # Initialize iterator with the test dataset
    sess.run(test_init_op)
    for i in range(test_batches_per_epoch):

        batch_xs, fnames = sess.run(next_batch)
        prediction = sess.run(pred,
                              feed_dict={
                                  X: batch_xs,
                                  mode: False}
                              )

        # Transform propabilities into binary values 0 or 1.
        test_pred = trsf_proba_to_binary(prediction)

        for i in range(len(test_pred)):
            imageId = fnames[i].decode()
            height, width = get_image_size(imageId)

            # Resize predicted masks to original image size.
            res_mask = trsf_proba_to_binary(
                resize(np.squeeze(test_pred[i]), (height, width), mode='constant', preserve_range=True)
            )
            #
            # fill the holes that remained
            res_mask = morpho_op(res_mask)
            # Rescale to 0-255 and convert to uint8
            res_mask = (255.0 * res_mask).astype(np.uint8)
            #
            test_pred_to_original_size = np.array(res_mask)

            # Run length encoding of predicted test masks.
            test_pred_rle = []
            test_pred_ids = []

            # calculate the minimum object size
            min_object_size = 20 * height * width / (256 * 256)

            # rle
            rle = list(mask_to_rle(test_pred_to_original_size, min_object_size=min_object_size))
            test_pred_rle.extend(rle)
            test_pred_ids.extend([imageId] * len(rle))

            sub = pd.DataFrame()
            sub['ImageId'] = test_pred_ids
            sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(lambda x: ' '.join(str(y) for y in x))

            if not os.path.isfile(filename):
                sub.to_csv(filename, index=False)
            else:
                sub.to_csv(filename, index=False, header=False, mode='a')

            process_count += 1
            print('evaluation... %d / %d' % (process_count, total_process_count))

    # add bulk data for invalid image
    sub = pd.DataFrame()
    sub['ImageId'] = ['5390acefd575cf9b33413ddf6cbb9ce137ae07dc04616ba24c7b5fe476c827d2']
    sub['EncodedPixels'] = pd.Series([[1, 1]]).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(filename, index=False, header=False, mode='a')

    end_time = datetime.datetime.now()
    print('end test: {}'.format(test_data.data_size, end_time))
    print('test waste time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        # default='/home/ace19/dl-data/nucleus_detection/stage1_train',
        # default='../../dl_data/nucleus/stage1_test',
        default='../../dl_data/nucleus/stage2_test_final',
        type=str,
        help="Data directory")

    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help="Batch size")

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.getcwd() + '/models',
        help='Directory to read checkpoint.')

    parser.add_argument(
        '--checkpoint_file',
        type=str,
        # default='unet.ckpt-20',
        default=None,
        help='checkpoint file name.')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.getcwd() + '/result',
        help='Directory to write submission.csv file.')

    parser.add_argument(
        '--img_size',
        type=int,
        default=512,
        # default=256,
        help="Image height and width")

    parser.add_argument(
        '--gpu_index',
        type=str,
        # default='0',
        default=None,
        help="Set the gpu index. If you not sepcify then auto")

    parser.add_argument(
        '--use_64_channel',
        type=bool,
        default=True,
        # default=False,
        help="If you set True then use the Unet_64_1024. otherwise use the Unet_32_512")

    parser.add_argument(
        '--conv_padding',
        type=str,
        default='same',
        # default='valid',
        help="conv padding. if your img_size is 572 and, conv_padding is valid then the label_size is 388")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)