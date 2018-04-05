
import tensorflow as tf

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D


L2_REG = 0.1

def _conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, filters in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                filters,
                (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def _upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = _upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def _upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
        name="upsample_{}".format(name))


def Unet_64_1024(X, training, flags=None):  # 512 x 512
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C) - 512 x 512
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    conv1, pool1 = _conv_conv_pool(X, [64, 64], training, flags, name=1)
    conv2, pool2 = _conv_conv_pool(pool1, [128, 128], training, flags, name=2)
    conv3, pool3 = _conv_conv_pool(pool2, [256, 256], training, flags, name=3)
    conv4, pool4 = _conv_conv_pool(pool3, [512, 512], training, flags, name=4)
    conv5 = _conv_conv_pool(pool4, [1024, 1024], training, flags, name=5, pool=False)

    up6 = _upconv_concat(conv5, conv4, 512, flags, name=6)
    conv6 = _conv_conv_pool(up6, [512, 512], training, flags, name=6, pool=False)

    up7 = _upconv_concat(conv6, conv3, 256, flags, name=7)
    conv7 = _conv_conv_pool(up7, [256, 256], training, flags, name=7, pool=False)

    up8 = _upconv_concat(conv7, conv2, 128, flags, name=8)
    conv8 = _conv_conv_pool(up8, [128, 128], training, flags, name=8, pool=False)

    up9 = _upconv_concat(conv8, conv1, 64, flags, name=9)
    conv9 = _conv_conv_pool(up9, [64, 64], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def Unet_32_512(X, training, flags=None): # 256 x 256
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C) - 256 x 256
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    conv1, pool1 = _conv_conv_pool(X, [32, 32], training, flags, name=1)
    conv2, pool2 = _conv_conv_pool(pool1, [64, 64], training, flags, name=2)
    conv3, pool3 = _conv_conv_pool(pool2, [128, 128], training, flags, name=3)
    conv4, pool4 = _conv_conv_pool(pool3, [256, 256], training, flags, name=4)
    conv5 = _conv_conv_pool(pool4, [512, 512], training, flags, name=5, pool=False)

    up6 = _upconv_concat(conv5, conv4, 256, flags, name=6)
    conv6 = _conv_conv_pool(up6, [256, 256], training, flags, name=6, pool=False)

    up7 = _upconv_concat(conv6, conv3, 128, flags, name=7)
    conv7 = _conv_conv_pool(up7, [128, 128], training, flags, name=7, pool=False)

    up8 = _upconv_concat(conv7, conv2, 64, flags, name=8)
    conv8 = _conv_conv_pool(up8, [64, 64], training, flags, name=8, pool=False)

    up9 = _upconv_concat(conv8, conv1, 32, flags, name=9)
    conv9 = _conv_conv_pool(up9, [32, 32], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')
