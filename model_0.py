# coding:utf-8

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers as tflayers


def model(_x, _y, batch_size, image_size):
    regularizer = tflayers.l2_regularizer(0.01, "regularizer")
    h_0 = tf.divide(tf.cast(_x, tf.float32), 255.)
    h_1 = tflayers.conv2d(h_0, 20, 5, 2, activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_1")
    h_2 = tflayers.conv2d(h_1, 40, 5, 2, activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_2")
    h_3 = tflayers.conv2d(h_2, 60, 5, 2, activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_3")
    h_4 = tflayers.conv2d(h_3, 80, 5, 2, activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_4")
    h_5 = tflayers.conv2d(unpool_2x2(h_4, "unpool_5"), 80, 5, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_5")
    h_6 = tflayers.conv2d(unpool_2x2(h_5, "unpool_6"), 40, 5, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_6")
    h_7 = tflayers.conv2d(unpool_2x2(h_6, "unpool_7"), 60, 5, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_7")
    h_8 = tflayers.conv2d(unpool_2x2(h_7, "unpool_8"), 20, 5, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_8")
    h_9 = tflayers.conv2d(h_8, 10, 5, 1, activation_fn=tf.nn.tanh, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer, scope="h_9")
    h_10 = tflayers.conv2d(h_9, 5, 5, 1, activation_fn=None, normalizer_fn=None, weights_regularizer=regularizer, scope="h_10")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_10, labels=tf.one_hot(_y, 5))) + tflayers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    predict = tf.cast(tf.argmax(h_10, axis=3), tf.uint8)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, _y), tf.float32) * tf.cast(tf.not_equal(_y, 0), tf.float32)) / tf.reduce_sum(tf.cast(tf.not_equal(_y, 0), tf.float32))
    return loss, accuracy, predict


def train(_batch_size, _block_size, step_start=0, restore=False):
    m_x, m_y = read_data("data/train_128_4/", _block_size)
    m_x, m_y = tf.train.shuffle_batch([m_x, m_y], batch_size=_batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(m_x, m_y, _batch_size, _block_size)
    tf.summary.scalar("loss", m_loss)
    tf.summary.scalar("accuracy", m_accuracy)
    merged_summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter("log", sess.graph)

    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "./model/model.ckpt-%d" % step_start)
    for step in range(step_start, 100000):
        if step % 100 == 0:
            v_accuracy, origin, marking = sess.run([m_accuracy, m_x, m_predict])
            save_sample(origin, marking, "sample/sample_%05d.png" % step)
            print("step %6d: acc = %f" % (step, v_accuracy))
            saver.save(sess, "./model/model.ckpt", global_step=step)
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: step})
        if step % 10 == 0:
            print("step %6d: loss = %f" % (step, v_loss))
            summary_writer.add_summary(sess.run(merged_summary_op), step)


if __name__ == '__main__':
    batch_size = 64
    block_size = 128
    train(batch_size, block_size, 0, False)
