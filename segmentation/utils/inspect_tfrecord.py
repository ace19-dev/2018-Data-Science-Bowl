import argparse
import sys

import tensorflow as tf


FLAGS = None

def print_tfrecords_count(tfrecord_filename):
    print('sum : ', sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_filename)))


# get a list of feature
def get_tfrecords_feature_list(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        return example.features.feature.keys()

    return []


# display record
def read_tfrecords(tfrecords_filename, is_train_val=False):
    ptr = 0
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img = (example.features.feature['image'].bytes_list.value[0])

        if is_train_val:
            label = (example.features.feature['label'].int64_list.value[0])
            yield ptr, img, label
        else:
            yield ptr, img

        ptr += 1

# save to file
def save_to_file():
    with open('train/labels_coco.txt', 'w') as f_labels:
        for idx, img, label in read_tfrecords('./train.tfrecords', is_train_val=True):
            fn = 'train/{}.png'.format(idx)
            with open(fn, 'wb') as f_img:
                f_img.write(img)
        print >>f_labels, "{} {}".format(fn, label)




def main(_):
    tfrecords_filename = FLAGS.data_dir + '/train_003.tfrecord'
    # get_tfrecords_feature_list(tfrecords_filename)
    print_tfrecords_count(tfrecords_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/ace19/dl-data/nucleus_detection/stage1_train_tfrecord',
        help='')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)