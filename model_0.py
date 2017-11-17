# coding:utf-8

import os
import numpy as np
from PIL import Image
import tensorflow as tf


def read_data():
    num = len(os.listdir("data/train")) // 2
    data = np.zeros(shape=[num, 128, 128, 3])
    label = np.zeros(shape=[num, 128, 128])
    for i in range(num):
        data[i, :, :, :] = np.array(Image.open("data/train/%06d.png" % i))
        label[i, :, :] = np.array(Image.open("data/train/%06d_class.png" % i))
    return data, label


def model():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="x")
    y = tf.placeholder(dtype=tf.int64, shape=[None, 128, 128], name="y")
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
    y_ = tf.nn.softmax(h3, dim=3)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h3, axis=3), y), tf.float32))
    return x, y, loss, acc, y_


batch_size = 64
X, Y = read_data()
X = X.astype(np.float32) / 255.
m_x, m_y, m_loss, m_acc, m_y_ = model()
op = tf.train.AdamOptimizer(0.01).minimize(m_loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
index = 0
for s in range(10000):
    _, v_loss = sess.run([op, m_loss], feed_dict={m_x: X[index:index + batch_size], m_y: Y[index:index + batch_size]})
    print("step %6d: loss = %f" % (s, v_loss))
    if s % 10 == 0:
        v_acc = sess.run(m_acc, feed_dict={m_x: X[index:index + batch_size], m_y: Y[index:index + batch_size]})
        print("step %6d: acc = %f" % (s, v_acc))
