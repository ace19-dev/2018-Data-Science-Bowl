
import tensorflow as tf

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


def Unet(X, training, flags=None):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    conv1, pool1 = _conv_conv_pool(X, [8, 8], training, flags, name=1)
    conv2, pool2 = _conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = _conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = _conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5, pool5 = _conv_conv_pool(pool4, [128, 128], training, flags, name=5)
    conv6, pool6 = _conv_conv_pool(pool5, [256, 256], training, flags, name=6)
    conv7 = _conv_conv_pool(pool6, [512, 512], training, flags, name=7, pool=False)

    up8 = _upconv_concat(conv7, conv6, 256, flags, name=8)
    conv8 = _conv_conv_pool(up8, [256, 256], training, flags, name=8, pool=False)

    up9 = _upconv_concat(conv8, conv5, 128, flags, name=9)
    conv9 = _conv_conv_pool(up9, [128, 128], training, flags, name=9, pool=False)

    up10 = _upconv_concat(conv9, conv4, 64, flags, name=10)
    conv10 = _conv_conv_pool(up10, [64, 64], training, flags, name=10, pool=False)

    up11 = _upconv_concat(conv10, conv3, 32, flags, name=11)
    conv11 = _conv_conv_pool(up11, [32, 32], training, flags, name=11, pool=False)

    up12 = _upconv_concat(conv11, conv2, 16, flags, name=12)
    conv12 = _conv_conv_pool(up12, [16, 16], training, flags, name=12, pool=False)

    up13 = _upconv_concat(conv12, conv1, 8, flags, name=13)
    conv13 = _conv_conv_pool(up13, [8, 8], training, flags, name=13, pool=False)

    return tf.layers.conv2d(
        conv13,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')




# def weight_variable(self, shape, name=None):
#     """ Weight initialization """
#     # initializer = tf.truncated_normal(shape, stddev=0.1)
#     initializer = tf.contrib.layers.xavier_initializer()
#     # initializer = tf.contrib.layers.variance_scaling_initializer()
#     return tf.get_variable(name, shape=shape, initializer=initializer)
#
# def bias_variable(self, shape, name=None):
#     """Bias initialization."""
#     # initializer = tf.constant(0.1, shape=shape)
#     initializer = tf.contrib.layers.xavier_initializer()
#     # initializer = tf.contrib.layers.variance_scaling_initializer()
#     return tf.get_variable(name, shape=shape, initializer=initializer)
#
# def conv2d(self, x, W, name=None):
#     """ 2D convolution. """
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
#
# def max_pool_2x2(self, x, name=None):
#     """ Max Pooling 2x2. """
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
#                           name=name)
#
# def conv2d_transpose(self, x, filters, name=None):
#     """ Transposed 2d convolution. """
#     return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=2,
#                                       strides=2, padding='SAME')
#
# def leaky_relu(self, z, name=None):
#     """Leaky ReLU."""
#     return tf.maximum(0.01 * z, z, name=name)
#
# def activation(self, x, name=None):
#     """ Activation function. """
#     a = tf.nn.elu(x, name=name)
#     # a = self.leaky_relu(x, name=name)
#     # a = tf.nn.relu(x, name=name)
#     return a
#
# def batch_norm_layer(self, x, name=None):
#     """Batch normalization layer."""
#     if False:
#         layer = tf.layers.batch_normalization(x, training=self.training_tf,
#                                               momentum=0.9, name=name)
#     else:
#         layer = x
#     return layer
#
# def dropout_layer(self, x, name=None):
#     """Dropout layer."""
#     if False:
#         layer = tf.layers.dropout(x, self.dropout_proba, training=self.training_tf,
#                                   name=name)
#     else:
#         layer = x
#     return layer
#
# def Unet(X, training, flags=None):
#     # 1. unit
#     with tf.name_scope('1.unit'):
#         W1_1 = weight_variable([3, 3, 3, 16], 'W1_1')
#         b1_1 = bias_variable([16], 'b1_1')
#         Z1 = conv2d(X, W1_1, 'Z1') + b1_1
#         A1 = activation(batch_norm_layer(Z1))  # (.,128,128,16)
#         A1_drop = dropout_layer(A1)
#         W1_2 = weight_variable([3, 3, 16, 16], 'W1_2')
#         b1_2 = bias_variable([16], 'b1_2')
#         Z2 = conv2d(A1_drop, W1_2, 'Z2') + b1_2
#         A2 = activation(batch_norm_layer(Z2))  # (.,128,128,16)
#         P1 = max_pool_2x2(A2, 'P1')  # (.,64,64,16)
#     # 2. unit
#     with tf.name_scope('2.unit'):
#         W2_1 = weight_variable([3, 3, 16, 32], "W2_1")
#         b2_1 = bias_variable([32], 'b2_1')
#         Z3 = conv2d(P1, W2_1) + b2_1
#         A3 = activation(batch_norm_layer(Z3))  # (.,64,64,32)
#         A3_drop = dropout_layer(A3)
#         W2_2 = weight_variable([3, 3, 32, 32], "W2_2")
#         b2_2 = bias_variable([32], 'b2_2')
#         Z4 = conv2d(A3_drop, W2_2) + b2_2
#         A4 = activation(batch_norm_layer(Z4))  # (.,64,64,32)
#         P2 = max_pool_2x2(A4)  # (.,32,32,32)
#     # 3. unit
#     with tf.name_scope('3.unit'):
#         W3_1 = weight_variable([3, 3, 32, 64], "W3_1")
#         b3_1 = bias_variable([64], 'b3_1')
#         Z5 = conv2d(P2, W3_1) + b3_1
#         A5 = activation(batch_norm_layer(Z5))  # (.,32,32,64)
#         A5_drop = dropout_layer(A5)
#         W3_2 = weight_variable([3, 3, 64, 64], "W3_2")
#         b3_2 = bias_variable([64], 'b3_2')
#         Z6 = conv2d(A5_drop, W3_2) + b3_2
#         A6 = activation(batch_norm_layer(Z6))  # (.,32,32,64)
#         P3 = max_pool_2x2(A6)  # (.,16,16,64)
#     # 4. unit
#     with tf.name_scope('4.unit'):
#         W4_1 = weight_variable([3, 3, 64, 128], "W4_1")
#         b4_1 = bias_variable([128], 'b4_1')
#         Z7 = conv2d(P3, W4_1) + b4_1
#         A7 = activation(batch_norm_layer(Z7))  # (.,16,16,128)
#         A7_drop = dropout_layer(A7)
#         W4_2 = weight_variable([3, 3, 128, 128], "W4_2")
#         b4_2 = bias_variable([128], 'b4_2')
#         Z8 = conv2d(A7_drop, W4_2) + b4_2
#         A8 = activation(batch_norm_layer(Z8))  # (.,16,16,128)
#         P4 = max_pool_2x2(A8)  # (.,8,8,128)
#     # 5. unit
#     with tf.name_scope('5.unit'):
#         W5_1 = weight_variable([3, 3, 128, 256], "W5_1")
#         b5_1 = bias_variable([256], 'b5_1')
#         Z9 = conv2d(P4, W5_1) + b5_1
#         A9 = activation(batch_norm_layer(Z9))  # (.,8,8,256)
#         A9_drop = dropout_layer(A9)
#         W5_2 = weight_variable([3, 3, 256, 256], "W5_2")
#         b5_2 = bias_variable([256], 'b5_2')
#         Z10 = conv2d(A9_drop, W5_2) + b5_2
#         A10 = activation(batch_norm_layer(Z10))  # (.,8,8,256)
#     # 6. unit
#     with tf.name_scope('6.unit'):
#         W6_1 = weight_variable([3, 3, 256, 128], "W6_1")
#         b6_1 = bias_variable([128], 'b6_1')
#         U1 = conv2d_transpose(A10, 128)  # (.,16,16,128)
#         U1 = tf.concat([U1, A8], 3)  # (.,16,16,256)
#         Z11 = conv2d(U1, W6_1) + b6_1
#         A11 = activation(batch_norm_layer(Z11))  # (.,16,16,128)
#         A11_drop = dropout_layer(A11)
#         W6_2 = weight_variable([3, 3, 128, 128], "W6_2")
#         b6_2 = bias_variable([128], 'b6_2')
#         Z12 = conv2d(A11_drop, W6_2) + b6_2
#         A12 = activation(batch_norm_layer(Z12))  # (.,16,16,128)
#     # 7. unit
#     with tf.name_scope('7.unit'):
#         W7_1 = weight_variable([3, 3, 128, 64], "W7_1")
#         b7_1 = bias_variable([64], 'b7_1')
#         U2 = conv2d_transpose(A12, 64)  # (.,32,32,64)
#         U2 = tf.concat([U2, A6], 3)  # (.,32,32,128)
#         Z13 = conv2d(U2, W7_1) + b7_1
#         A13 = activation(batch_norm_layer(Z13))  # (.,32,32,64)
#         A13_drop = dropout_layer(A13)
#         W7_2 = weight_variable([3, 3, 64, 64], "W7_2")
#         b7_2 = bias_variable([64], 'b7_2')
#         Z14 = conv2d(A13_drop, W7_2) + b7_2
#         A14 = activation(batch_norm_layer(Z14))  # (.,32,32,64)
#     # 8. unit
#     with tf.name_scope('8.unit'):
#         W8_1 = weight_variable([3, 3, 64, 32], "W8_1")
#         b8_1 = bias_variable([32], 'b8_1')
#         U3 = conv2d_transpose(A14, 32)  # (.,64,64,32)
#         U3 = tf.concat([U3, A4], 3)  # (.,64,64,64)
#         Z15 = conv2d(U3, W8_1) + b8_1
#         A15 = activation(batch_norm_layer(Z15))  # (.,64,64,32)
#         A15_drop = dropout_layer(A15)
#         W8_2 = weight_variable([3, 3, 32, 32], "W8_2")
#         b8_2 = bias_variable([32], 'b8_2')
#         Z16 = conv2d(A15_drop, W8_2) + b8_2
#         A16 = activation(batch_norm_layer(Z16))  # (.,64,64,32)
#     # 9. unit
#     with tf.name_scope('9.unit'):
#         W9_1 = weight_variable([3, 3, 32, 16], "W9_1")
#         b9_1 = bias_variable([16], 'b9_1')
#         U4 = conv2d_transpose(A16, 16)  # (.,128,128,16)
#         U4 = tf.concat([U4, A2], 3)  # (.,128,128,32)
#         Z17 = conv2d(U4, W9_1) + b9_1
#         A17 = activation(batch_norm_layer(Z17))  # (.,128,128,16)
#         A17_drop = dropout_layer(A17)
#         W9_2 = weight_variable([3, 3, 16, 16], "W9_2")
#         b9_2 = bias_variable([16], 'b9_2')
#         Z18 = conv2d(A17_drop, W9_2) + b9_2
#         A18 = activation(batch_norm_layer(Z18))  # (.,128,128,16)
#     # 10. unit: output layer
#     with tf.name_scope('10.unit'):
#         W10 = weight_variable([1, 1, 16, 1], "W10")
#         b10 = bias_variable([1], 'b10')
#         Z19 = conv2d(A18, W10) + b10
#         A19 = tf.nn.sigmoid(batch_norm_layer(Z19))  # (.,128,128,1)
#
#     return A19
