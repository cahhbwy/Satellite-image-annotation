# coding:utf-8

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from preprocess import visualization


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


def model(x, y):
    with tf.variable_scope("conv2d_1"):
        w_1 = tf.get_variable("weights", shape=[5, 5, 3, 16], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32))
        b_1 = tf.get_variable("bias", shape=[16], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, dtype=tf.float32))
    h1 = tf.nn.bias_add(tf.nn.conv2d(x, filter=w_1, strides=[1, 1, 1, 1], padding="SAME"), b_1)
    with tf.variable_scope("conv2d_2"):
        w_2 = tf.get_variable("weights", shape=[5, 5, 16, 32], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32))
        b_2 = tf.get_variable("bias", shape=[32], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, dtype=tf.float32))
    h2 = tf.nn.bias_add(tf.nn.conv2d(h1, filter=w_2, strides=[1, 1, 1, 1], padding="SAME"), b_2)
    with tf.variable_scope("conv2d_3"):
        w_3 = tf.get_variable("weights", shape=[5, 5, 32, 5], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32))
        b_3 = tf.get_variable("bias", shape=[5], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, dtype=tf.float32))
    h3 = tf.nn.bias_add(tf.nn.conv2d(h2, filter=w_3, strides=[1, 1, 1, 1], padding="SAME"), b_3)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=tf.one_hot(y, 5)))
    predict = tf.cast(tf.argmax(h3, axis=3), tf.uint8)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(h3, axis=3), tf.uint8), y), tf.float32))
    return loss, acc, predict


if __name__ == '__main__':
    batch_size = 16
    X, Y = read_data()
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    X = tf.divide(tf.cast(X, tf.float32), 255.)
    m_loss, m_acc, m_predict = model(X, Y)
    op = tf.train.AdamOptimizer(0.01).minimize(m_loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    index = 0
    for s in range(10000):
        _, v_loss = sess.run([op, m_loss])
        print("step %6d: loss = %f" % (s, v_loss))
        if s % 10 == 0:
            v_acc = sess.run(m_acc)
            print("step %6d: acc = %f" % (s, v_acc))
