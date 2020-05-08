# coding:utf-8
# Wassertein GANs

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generator(_x, _rand_dim, _batch_size, reuse=False):
    with tf.name_scope("generator"):
        shape = [s.value for s in _x.get_shape()]
        shape[3] = _rand_dim
        g_0 = tf.concat([tf.divide(tf.cast(_x, tf.float32), 255.), tf.random_uniform(shape, 0.0, 1.0)], axis=3)
        g_1 = tflayers.conv2d(g_0, 30, 11, 1, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="gen_1")
        g_2 = tflayers.conv2d(g_1, 30, 11, 1, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="gen_2")
        g_3 = tflayers.conv2d(g_2, 5, 11, 1, activation_fn=None, reuse=reuse, scope="gen_4")
        return g_3


def discriminator(_origin, _marking_onehot, _batch_size, reuse=False):
    with tf.name_scope("discriminator"):
        d_0 = tf.concat([_origin, _marking_onehot], axis=3, name="dis_concat_0")
        d_1 = tflayers.conv2d(d_0, 16, 15, 4, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="dis_1")  # 128 -> 32
        d_2 = tflayers.conv2d(d_1, 32, 7, 2, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="dis_2")  # 32 -> 16
        d_3 = tflayers.conv2d(d_2, 64, 5, 2, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="dis_3")  # 16 -> 8
        d_4 = tflayers.conv2d(d_3, 128, 3, 2, activation_fn=tf.nn.leaky_relu, reuse=reuse, scope="dis_4")  # 8 -> 4
        d_5 = tflayers.fully_connected(tf.reshape(d_4, [_batch_size, 128 * 4 * 4]), 512, activation_fn=tf.nn.tanh, reuse=reuse, scope="dis_5")
        d_6 = tflayers.fully_connected(d_5, 32, activation_fn=None, reuse=reuse, scope="dis_6")
        return d_6


def predict_loss(_o, _y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_o, labels=tf.one_hot(_y, 5)))
    predict = tf.cast(tf.argmax(_o, axis=3), tf.uint8)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, _y), tf.float32) * tf.cast(tf.not_equal(_y, 0), tf.float32)) / tf.reduce_sum(tf.cast(tf.not_equal(_y, 0), tf.float32))
    return loss, accuracy


def model(_origin_uint8, _marking, _batch_size, _rand_dim):
    origin = tf.divide(tf.cast(_origin_uint8, tf.float32), 255.)
    real_marking = tf.one_hot(_marking, 5)
    fake_marking = generator(origin, _rand_dim, _batch_size)
    real = discriminator(origin, real_marking, _batch_size)
    fake = discriminator(origin, fake_marking, _batch_size, reuse=True)
    loss, accuracy = predict_loss(fake_marking, _marking)
    dis_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    gen_loss = -tf.reduce_mean(fake)
    alpha = tf.random_uniform(shape=[_batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_marking + (1. - alpha) * fake_marking
    gradients = tf.gradients(discriminator(origin, interpolates, _batch_size, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    dis_loss += 10 * gradient_penalty
    return dis_loss, gen_loss, loss, accuracy


def sample_generator(_origin, _rand_dim, _batch_size):
    s_0 = generator(_origin, _rand_dim, _batch_size, True)
    return tf.cast(tf.argmax(s_0, axis=3), tf.uint8)


def train(_batch_size, _rand_dim):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=global_step, decay_steps=100, decay_rate=0.90)
    origin, real_marking = read_data("data/train_128_8/", 128)
    origin, real_marking = tf.train.shuffle_batch([origin, real_marking], batch_size=_batch_size, capacity=2000, min_after_dequeue=1000)
    m_dis_loss, m_gen_loss, m_loss, m_accuracy = model(origin, real_marking, _batch_size, _rand_dim)
    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]
    train_op_dis = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(m_dis_loss, var_list=dis_vars)
    train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(m_gen_loss, var_list=gen_vars)
    sample = sample_generator(origin, _rand_dim, _batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    for step in range(10000):
        v_dis_loss, v_gen_loss = 0.0, 0.0
        if step % 100 == 0:
            v_origin, v_fake_marking, v_accuracy = sess.run([origin, sample, m_accuracy])
            save_sample(v_origin, v_fake_marking, "sample/sample_%05d.png" % step)
            print("step: %5d; accuracy: %6f" % (step, v_accuracy))
        for _ in range(1):
            _, v_dis_loss = sess.run([train_op_dis, m_dis_loss])
        for _ in range(3):
            _, v_gen_loss = sess.run([train_op_gen, m_gen_loss])
        if step % 10 == 0:
            print("step: %5d; dis_loss: %6f, gen_loss: %6f" % (step, v_dis_loss, v_gen_loss))


if __name__ == '__main__':
    train(64, 16)
