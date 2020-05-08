# coding:utf-8
# 卷积神经网络，金字塔

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers as tflayers


def model(_x, _y, _batch_size, _image_size):
    regularizer = tflayers.l2_regularizer(0.1, "regularizer")
    x_00 = tf.divide(tf.cast(_x, tf.float32), 255.)
    y_00 = tf.reshape(_y, [_batch_size, _image_size, _image_size, 1])
    s = [(_image_size // i, _image_size // i) for i in [16, 8, 4, 2, 1]]
    _loss = [0.0] * 5

    h_10 = tf.image.resize_bilinear(x_00, s[0])
    y_10 = tf.reshape(tf.image.resize_nearest_neighbor(y_00, s[0]), [_batch_size, s[0][0], s[0][1]])
    h_11 = tflayers.conv2d(h_10, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_12 = tflayers.conv2d(h_11, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_13 = tflayers.conv2d(h_12, 5, 3, 1, activation_fn=None, weights_regularizer=regularizer)
    _loss[0] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_13, labels=tf.one_hot(y_10, 5)))

    h_20 = tf.image.resize_bilinear(h_13, s[1])
    y_20 = tf.reshape(tf.image.resize_nearest_neighbor(y_00, s[1]), [_batch_size, s[1][0], s[1][1]])
    h_21 = tflayers.conv2d(h_20, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_22 = tflayers.conv2d(h_21, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_23 = tflayers.conv2d(h_22, 5, 3, 1, activation_fn=None, weights_regularizer=regularizer)
    _loss[1] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_23, labels=tf.one_hot(y_20, 5)))

    h_30 = tf.image.resize_bilinear(h_23, s[2])
    y_30 = tf.reshape(tf.image.resize_nearest_neighbor(y_00, s[2]), [_batch_size, s[2][0], s[2][1]])
    h_31 = tflayers.conv2d(h_30, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_32 = tflayers.conv2d(h_31, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_33 = tflayers.conv2d(h_32, 5, 3, 1, activation_fn=None, weights_regularizer=regularizer)
    _loss[2] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_33, labels=tf.one_hot(y_30, 5)))

    h_40 = tf.image.resize_bilinear(h_33, s[3])
    y_40 = tf.reshape(tf.image.resize_nearest_neighbor(y_00, s[3]), [_batch_size, s[3][0], s[3][1]])
    h_41 = tflayers.conv2d(h_40, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_42 = tflayers.conv2d(h_41, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_43 = tflayers.conv2d(h_42, 5, 3, 1, activation_fn=None, weights_regularizer=regularizer)
    _loss[3] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_43, labels=tf.one_hot(y_40, 5)))

    h_50 = tf.image.resize_bilinear(h_43, s[4])
    y_50 = tf.reshape(tf.image.resize_nearest_neighbor(y_00, s[4]), [_batch_size, s[4][0], s[4][1]])
    h_51 = tflayers.conv2d(h_50, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_52 = tflayers.conv2d(h_51, 24, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.batch_norm, weights_regularizer=regularizer)
    h_53 = tflayers.conv2d(h_52, 5, 3, 1, activation_fn=None, weights_regularizer=regularizer)
    _loss[4] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=h_53, labels=tf.one_hot(y_50, 5)))

    loss = tf.reduce_sum(_loss) + tflayers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    predict = tf.cast(tf.argmax(h_53, axis=3), tf.uint8)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, _y), tf.float32) * tf.cast(tf.not_equal(_y, 0), tf.float32)) / tf.reduce_sum(tf.cast(tf.not_equal(_y, 0), tf.float32))

    return loss, accuracy, predict


def train(_batch_size, _block_size, step_start=0, restore=False):
    m_x, m_y = read_data("data/train_128_8/", _block_size)
    m_x, m_y = tf.train.shuffle_batch([m_x, m_y], batch_size=_batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(m_x, m_y, _batch_size, _block_size)
    tf.summary.scalar("loss", m_loss)
    tf.summary.scalar("accuracy", m_accuracy)
    merged_summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
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
