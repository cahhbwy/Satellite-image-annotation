# coding:utf-8

import os
from PIL import Image
import numpy as np
import tensorflow as tf


# 0. 其他 ( 64 , 64,  64)
# 1. 植被 (  0 ,128,   0)
# 2. 道路 (255 ,255,   0)
# 3. 建筑 (128 ,128, 128)
# 4. 水体 (  0 ,  0, 255)
def visualization(_origin, _marking, alpha=0.5, show=True, save=False, save_path="temp.png"):
    if not isinstance(_origin, np.ndarray):
        _origin = np.array(_origin)
    if not isinstance(_marking, np.ndarray):
        _marking = np.array(_marking)
    colors = [[64, 64, 64], [0, 128, 0], [255, 255, 0], [128, 128, 128], [0, 0, 255]]
    img = np.zeros(shape=(_marking.shape[0], _marking.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        x, y = np.where(_marking == i)
        img[x, y, :] = colors[i]
    if save:
        Image.fromarray((alpha * img + (1 - alpha) * _origin).astype(np.uint8)).save(save_path)
    if show:
        Image.fromarray((alpha * img + (1 - alpha) * _origin).astype(np.uint8)).show()


def split_data(_origin, _marking, block_size=256, overlapping=0.3, use_tf=False, image_num=1024, random=False, total=8192):
    shape = _marking.size
    if random:
        x = np.random.randint(0, shape[0] - block_size, total)
        y = np.random.randint(0, shape[1] - block_size, total)
        if use_tf:
            count = 0
            file_num = 0
            for i in range(total):
                if count == 0:
                    print("create data_%04d.tfrecords" % file_num)
                    writer = tf.python_io.TFRecordWriter("data/train/data_%04d.tfrecords" % file_num)
                origin_raw = _origin.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).tobytes()
                marking_raw = _marking.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[origin_raw])),
                    'marking': tf.train.Feature(bytes_list=tf.train.BytesList(value=[marking_raw]))
                }))
                writer.write(example.SerializeToString())
                count += 1
                if count >= image_num:
                    count = 0
                    file_num += 1
                    writer.close()
        else:
            for i in range(total):
                _origin.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).save("data/train/%06d.png" % i)
                _marking.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).save("data/train/%06d_class.png" % i)
    else:
        i = 0
        if use_tf:
            count = 0
            file_num = 0
            for x in range(0, shape[0] - block_size, int(block_size * (1 - overlapping))):
                for y in range(0, shape[1] - block_size, int(block_size * (1 - overlapping))):
                    if count == 0:
                        print("create data_%04d.tfrecords" % file_num)
                        writer = tf.python_io.TFRecordWriter("data/train/data_%04d.tfrecords" % file_num)
                    origin_raw = _origin.crop((x, y, x + block_size, y + block_size)).tobytes()
                    marking_raw = _marking.crop((x, y, x + block_size, y + block_size)).tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[origin_raw])),
                        'marking': tf.train.Feature(bytes_list=tf.train.BytesList(value=[marking_raw]))
                    }))
                    writer.write(example.SerializeToString())
                    count += 1
                    if count >= image_num:
                        count = 0
                        file_num += 1
                        writer.close()
        else:
            for x in range(0, shape[0] - block_size, int(block_size * (1 - overlapping))):
                for y in range(0, shape[1] - block_size, int(block_size * (1 - overlapping))):
                    _origin.crop((x, y, x + block_size, y + block_size)).save("data/train/%06d.png" % i)
                    _marking.crop((x, y, x + block_size, y + block_size)).save("data/train/%06d_class.png" % i)
                    i += 1


def read_data(dir_path="data/train/", block_size=256):
    file_list = [dir_path + filename for filename in os.listdir(dir_path)]
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example, features={
        'origin': tf.FixedLenFeature([], tf.string),
        'marking': tf.FixedLenFeature([], tf.string)
    })
    _origin = tf.decode_raw(features['origin'], tf.uint8)
    _marking = tf.decode_raw(features['marking'], tf.uint8)
    _origin = tf.reshape(_origin, (block_size, block_size, 3))
    _marking = tf.reshape(_marking, (block_size, block_size))
    return _origin, _marking


def save_sample(origin_ndarry, marking_ndarry, path="temp.png"):
    count = origin_ndarry.shape[0]
    size = origin_ndarry.shape[1]
    row = np.ceil(np.sqrt(count)).astype(np.int32)
    col = count // row
    img_h = row * size
    img_w = col * size
    _origin = np.zeros(shape=[img_h, img_w, 3], dtype=np.uint8)
    _marking = np.ones(shape=[img_h, img_w], dtype=np.uint8) * 255
    for i in range(count):
        x = (i // col) * size
        y = (i % col) * size
        _origin[x:x + size, y:y + size, :] = origin_ndarry[i]
        _marking[x:x + size, y:y + size] = marking_ndarry[i]
    visualization(_origin, _marking, show=False, save=True, save_path=path)


def bias(name, shape, bias_start=0.0, trainable=True):
    return tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.constant_initializer(bias_start, dtype=tf.float32))


def weight(name, shape, stddev=0.5, trainble=True):
    return tf.get_variable(name, shape, tf.float32, trainable=trainble, initializer=tf.random_normal_initializer(stddev=stddev, dtype=tf.float32))


def fully_connected(value, output_shape, name="fully_connected", with_w=False):
    shape = value.get_shape().as_list()
    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)
    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)


def deconv2d(value, output_shape, k_h=5, k_w=5, strides=(1, 2, 2, 1), name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=strides)
        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


def conv2d(value, output_dim, k_h=5, k_w=5, strides=(1, 2, 2, 1), name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def conv_cond_concat(value, cond, name='concat'):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    with tf.variable_scope(name):
        return tf.concat(axis=3, values=[value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])])


def batch_norm(value, is_train=True, name='batch_norm', epsilon=1e-5, momentum=0.9):
    with tf.variable_scope(name):
        ema = tf.train.ExponentialMovingAverage(decay=momentum)
        shape = value.get_shape().as_list()[-1]
        beta = bias('beta', [shape], bias_start=0.0)
        gamma = bias('gamma', [shape], bias_start=1.0)
        if is_train:
            batch_mean, batch_variance = tf.nn.moments(value, list(range(len(value.get_shape().as_list()) - 1)), name='moments')
            moving_mean = bias('moving_mean', [shape], 0.0, False)
            moving_variance = bias('moving_variance', [shape], 1.0, False)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema_apply_op = ema.apply([batch_mean, batch_variance])
            assign_mean = moving_mean.assign(ema.average(batch_mean))
            assign_variance = moving_variance.assign(ema.average(batch_variance))
            with tf.control_dependencies([ema_apply_op]):
                mean, variance = tf.identity(batch_mean), tf.identity(batch_variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(value, mean, variance, beta, gamma, 1e-5)
        else:
            mean = bias('moving_mean', [shape], 0.0, False)
            variance = bias('moving_variance', [shape], 1.0, False)
            return tf.nn.batch_normalization(value, mean, variance, beta, gamma, epsilon)


if __name__ == '__main__':
    marking = Image.open("data/CCF-training/1_class_8bits.png")
    origin = Image.open("data/CCF-training/1-8bits.png")

    # visualization(_origin, _marking, 0.5)
    split_data(origin, marking, use_tf=True, random=True)
