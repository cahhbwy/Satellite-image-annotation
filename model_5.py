# coding:utf-8
# accuarcy: 80%

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers as tflayers


def model(_x, _y, _batch_size, _block_size):
    regularizer = tflayers.l2_regularizer(0.01, "regularizer")
    h_00 = tf.divide(tf.cast(_x, tf.float32), 255.)
    h_11 = tflayers.conv2d(h_00, 8, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_11")
    h_12 = tflayers.conv2d(h_11, 16, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_12")
    h_13 = tflayers.max_pool2d(h_12, 2, 2, scope="layer_13")

    h_21 = tflayers.conv2d(h_13, 16, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_21")
    h_22 = tflayers.conv2d(h_21, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_22")
    h_23 = tflayers.max_pool2d(h_22, 2, 2, scope="layer_23")

    h_31 = tflayers.conv2d(h_23, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_31")
    h_32 = tflayers.conv2d(h_31, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_32")
    h_33 = tflayers.conv2d(h_32, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_33")
    h_34 = tflayers.max_pool2d(h_33, 2, 2, scope="layer_33")

    h_41 = tflayers.conv2d(h_34, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_41")
    h_42 = tflayers.conv2d(h_41, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_42")
    h_43 = tflayers.conv2d(h_42, 128, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_43")

    h_51 = tflayers.conv2d(h_43, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_51")
    h_52 = tflayers.conv2d(h_51, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_52")
    h_53 = tflayers.conv2d(h_52, 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_53")

    h_61 = unpool_2x2(h_53, "layer_61")
    h_62 = tflayers.conv2d(h_61, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_62")
    h_63 = tflayers.conv2d(h_62, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_63")
    h_64 = tflayers.conv2d(h_63, 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_64")

    h_71 = unpool_2x2(h_64, "layer_71")
    h_72 = tflayers.conv2d(h_71, 16, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_72")
    h_73 = tflayers.conv2d(h_72, 16, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_73")

    h_81 = unpool_2x2(h_73, "layer_81")
    h_82 = tflayers.conv2d(h_81, 8, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer, scope="layer_82")
    h_83 = tflayers.conv2d(h_82, 5, 3, 1, activation_fn=None, normalizer_fn=None, weights_regularizer=regularizer, scope="layer_83")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_83, labels=tf.one_hot(_y, 5))) + tf.contrib.layers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    predict = tf.cast(tf.argmax(h_83, axis=3), tf.uint8)
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
