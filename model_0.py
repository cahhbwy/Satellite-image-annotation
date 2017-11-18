# coding:utf-8

from utils import *
import tensorflow as tf


def model(x, y, batch_size, image_size):
    h_0 = tf.divide(tf.cast(x, tf.float32), 255.)
    h_1 = batch_norm(conv2d(h_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="conv2d_0"), name="bn_1")
    h_2 = tf.nn.relu(batch_norm(conv2d(h_1, 20, k_h=9, k_w=9, strides=(1, 1, 1, 1), name="conv2d_2"), name="bn_3"))
    h_3 = conv2d(h_2, 5, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_4")
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_3, labels=tf.one_hot(y, 5)))
    predict = tf.cast(tf.argmax(h_3, axis=3), tf.uint8)
    acc = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))
    return loss, acc, predict


if __name__ == '__main__':
    batch_size = 16
    X, Y = read_data()
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_acc, m_predict = model(X, Y, batch_size=batch_size, image_size=256)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=100, decay_rate=0.95)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    index = 0
    for s in range(10000):
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: s})
        if s % 10 == 0:
            print("step %6d: loss = %f" % (s, v_loss))
        if s % 100 == 0:
            v_acc, origin, marking = sess.run([m_acc, X, m_predict])
            save_sample(origin, marking, "data/sample/sample_%05d.png" % s)
            print("step %6d: acc = %f" % (s, v_acc))
