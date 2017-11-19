# coding:utf-8
# GAN模型

from utils import *
import numpy as np
import tensorflow as tf


def generator(_origin, _rand, _rand_dim, _batch_size, _is_train=True):
    with tf.name_scope("generator"):
        g_0 = tf.divide(tf.cast(_origin, tf.float32), 255.)
        g_1 = lrelu(conv2d(g_0, 30, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="gen_conv2d_1"))
        g_2 = lrelu(conv2d(g_1, 30, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="gen_conv2d_2"))
        g_3 = lrelu(conv2d(g_2, 30, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="gen_conv2d_3"))
        g_4 = conv2d(g_3, 5, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="gen_conv2d_6")
        return g_4


def discriminator(_origin, _marking_onehot, _batch_size, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        with tf.name_scope("discriminator"):
            # tf.get_variable_scope().reuse_variables()
            d_0 = tf.concat([_origin, _marking_onehot], axis=3, name="dis_concat_0")
            d_1 = lrelu(conv2d(d_0, 32, k_h=15, k_w=15, strides=(1, 4, 4, 1), name="dis_conv2d_1"))  # (64,64)
            d_2 = lrelu(conv2d(d_1, 16, k_h=7, k_w=7, strides=(1, 2, 2, 1), name="dis_conv2d_2"))  # (32,32)
            d_3 = lrelu(conv2d(d_2, 8, k_h=7, k_w=7, strides=(1, 2, 2, 1), name="dis_conv2d_3"))  # (16,16)
            d_4 = lrelu(conv2d(d_3, 16, k_h=7, k_w=7, strides=(1, 2, 2, 1), name="dis_conv2d_4"))  # (8,8)
            d_5 = lrelu(conv2d(d_4, 32, k_h=7, k_w=7, strides=(1, 2, 2, 1), name="dis_conv2d_5"))  # (4,4)
            d_6 = tf.reshape(d_5, [_batch_size, 32 * 4 * 4], name="dis_reshape_6")
            d_7 = tf.nn.tanh(fully_connected(d_6, 32, name="dis_fc_7"))
            d_8 = fully_connected(d_7, 1, name="dis_fc_8")
            return d_8


def predict_loss(_real_marking, _fake_marking):
    _loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=_fake_marking, labels=_real_marking))
    _accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(_fake_marking, axis=3), tf.uint8), tf.cast(tf.argmax(_real_marking, axis=3), tf.uint8)), tf.float32))
    return _loss, _accuracy


def model(_origin_uint8, _marking, _batch_size, _rand_dim):
    _origin = tf.divide(tf.cast(_origin_uint8, tf.float32), 255.)
    _rand = tf.placeholder(tf.float32, [None, _rand_dim])
    _real_marking = tf.one_hot(_marking, 5)
    _fake_marking = generator(_origin, _rand, _rand_dim, _batch_size)
    _real = discriminator(_origin, _real_marking, _batch_size)
    _fake = discriminator(_origin, _fake_marking, _batch_size, reuse=True)
    dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(_real), logits=_real))
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(_fake), logits=_fake))
    _dis_loss = dis_loss_real + dis_loss_fake
    _gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(_fake), logits=_fake))
    _loss, _accuracy = predict_loss(_real_marking, _fake_marking)
    return _rand, _dis_loss, _gen_loss, _loss, _accuracy


def sample_generator(_origin, _rand_dim, _batch_size, _is_train=False):
    with tf.name_scope("sample"):
        _rand = tf.placeholder(tf.float32, [None, _rand_dim])
        tf.get_variable_scope().reuse_variables()
        s_0 = generator(_origin, _rand, _rand_dim, _batch_size, _is_train)
        return _rand, tf.cast(tf.argmax(s_0, axis=3), tf.uint8)


if __name__ == '__main__':
    batch_size = 16
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=global_step, decay_steps=100, decay_rate=0.90)
    origin, real_marking = read_data()
    origin, real_marking = tf.train.shuffle_batch([origin, real_marking], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    rand_dim = 32
    rand, dis_loss, gen_loss, loss, accuracy = model(origin, real_marking, batch_size, rand_dim)

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        train_op_dis = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(dis_loss, var_list=dis_vars)
        train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(gen_loss, var_list=gen_vars)

    sample_rand, sample = sample_generator(origin, rand_dim, batch_size)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    for step in range(10000):
        dis_loss_val, gen_loss_val = 0.0, 0.0
        rand_val = np.random.uniform(-1, 1, size=(batch_size, rand_dim)).astype(np.float32)
        for _ in range(1):
            _, dis_loss_val = sess.run([train_op_dis, dis_loss], feed_dict={rand: rand_val})
        for _ in range(3):
            _, gen_loss_val = sess.run([train_op_gen, gen_loss], feed_dict={rand: rand_val})
        if step % 10 == 0:
            print("step: %5d; dis_loss: %6f, gen_loss: %6f" % (step, dis_loss_val, gen_loss_val))
        if step % 100 == 0:
            origin_val, fake_marking_val, accuracy_val = sess.run([origin, sample, accuracy], feed_dict={sample_rand: rand_val, rand: rand_val})
            save_sample(origin_val, fake_marking_val, "data/sample/sample_%05d.png" % step)
            print("step: %5d; accuracy: %6f" % (step, accuracy_val))
