# coding=utf-8
from __future__ import division, print_function
import tensorflow.compat.v1 as tf
import tf_slim as slim

"""
yolo网络络模块
"""

def conv2d(inputs, filters, kernel_size, strides=1):
    """
    net = conv2d(net, 64,  3, strides=2)
    卷积构建
    :param inputs: 输入
    :param filters: 卷积核数量（也就是输出的channels）
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :return:
    """
    log = ''

    def _zero_pading(inputs, kernel_size):
        """
        固定填充,不依赖输入大小
        :param inputs:
        :param kernel_size:
        :return:
        """

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(
            inputs,
            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]],
            mode='CONSTANT'
        )
        return padded_inputs

    if strides > 1:
        inputs = _zero_pading(inputs, kernel_size)
        log += "zero_padding--"

    out = slim.conv2d(
        inputs, filters, kernel_size, stride=strides,
        padding=('SAME' if strides == 1 else 'VALID')
    )
    log = log + "\033[0mconv--(name: %s kernel_size:%d*%d kernel_num:%d stride:%d " % (inputs.name, kernel_size,  kernel_size, filters, strides)
    # log = log + "input:%d*%d*%d " % (inputs.shape[1], inputs.shape[2], inputs.shape[3])
    # log = log + "out:%d*%d*%d " % (out.shape[1], out.shape[2], inputs.shape[3])
    print(log)
    return out


def res(inputs, filters):
    """
    残差块
    :param inputs: 输入
    :param filters: 过滤器数量
    :return:
    """
    print("res--")
    shortcut = inputs  # 输入副本
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = net + shortcut   # 拼接
    return net


def yolo_block(inputs, filters, class_num):
    """
    由上采样层、3个具有线性激活功能的卷积层，从而在3种不同的尺度上进行检测。
    :param inputs:
    :param filters: 卷积核数量
    :return:
    """
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    shortcut = net
    net = conv2d(net, filters * 2, 3)

    feature_map = slim.conv2d(
        net, 3 * (5 + class_num), 1,
        stride=1, normalizer_fn=None,
        activation_fn=None, biases_initializer=tf.zeros_initializer(),
    )
    return shortcut, feature_map


def upsample_layer(inputs, out_shape):
    """
    上采样：使用邻值插入
    :param inputs:
    :param out_shape:
    :return:
    """
    new_height, new_width = out_shape[1], out_shape[2]  # 先高再宽
    # TODO: Do we need to set `align_corners` as True?
    # 使用近邻值插入调整图像
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs


def darknet53(inputs):
    """
    darknet53网络主体，它包含53个卷积层
    每个卷后面都有BN层和Leaky ReLU激活层。
    下采样由带有stride = 2的conv层完成。
    :param inputs: input
    :return:
    """
    print("\033[32mBegin building Darknet-53...")
    # 第一层的两个卷积
    net = conv2d(inputs, 32,  3, strides=1)  # 3,3,1
    net = conv2d(net, 64,  3, strides=2)

    # res1
    net = res(net, 32)
    net = conv2d(net, 128, 3, strides=2)

    # res2
    for i in range(2):
        net = res(net, 64)
    net = conv2d(net, 256, 3, strides=2)

    # res8
    for i in range(8):
        net = res(net, 128)
    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res8
    for i in range(8):
        net = res(net, 256)
    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res4
    for i in range(4):
        net = res(net, 512)
    route_3 = net
    print("\033[32mFinish building Darknet-53...")
    return route_1, route_2, route_3


def detect_net(route_1, route_2, route_3, use_static_shape, class_num):
    """
    DarkNet-53输出的三个值
    :param route_1:
    :param route_2:
    :param route_3:
    :param use_static_shape:
    :param class_num:
    :return:
    """
    print("\033[32mBegin building detect net after Darknet-53...")
    inter1, feature_map_1 = yolo_block(route_3, 512, class_num)
    inter1 = conv2d(inter1, 256, 1)
    inter1 = upsample_layer(
        inter1, route_2.get_shape().as_list() if use_static_shape else tf.shape(route_2)
    )
    concat1 = tf.concat([inter1, route_2], axis=3)
    inter2, feature_map_2 = yolo_block(concat1, 256, class_num)

    inter2 = conv2d(inter2, 128, 1)
    inter2 = upsample_layer(
        inter2, route_1.get_shape().as_list() if use_static_shape else tf.shape(route_1)
    )
    concat2 = tf.concat([inter2, route_1], axis=3)
    _, feature_map_3 = yolo_block(concat2, 128, class_num)
    print("\033[32mFinishe building detect net after Darknet-53 end...")
    return feature_map_1, feature_map_2, feature_map_3
