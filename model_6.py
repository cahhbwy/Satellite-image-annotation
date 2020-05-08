# coding:utf-8
# accuarcy: 75%

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers as tflayers


def block_conv2d(_input, _block_size, num_outputs, kernel_size, stride, activation_fn, normalizer_fn, regularizer, scope):
    shape = [s.value for s in _input.get_shape()]
    reuse = None
    output = []
    for x in range(0, shape[1], _block_size):
        buffer = []
        for y in range(0, shape[2], _block_size):
            block = tf.slice(_input, [0, x, y, 0], [shape[0], _block_size, _block_size, shape[3]])
            buffer.append(tflayers.conv2d(block, num_outputs, kernel_size, stride, activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                                          weights_regularizer=regularizer,  reuse=reuse, scope=scope))
            reuse = True
        output.append(tf.concat(buffer, 2))
    return tf.concat(output, 1)


def model(_x, _y, _batch_size, _block_size):
    regularizer = tflayers.l2_regularizer(0.01, "regularizer")
    h_0 = tf.divide(tf.cast(_x, tf.float32), 255.)
    h_1 = block_conv2d(h_0, 128, 16, 3, 2, tf.nn.leaky_relu, tflayers.layer_norm, regularizer, "h_1")  # 512 -> 256
    h_2 = block_conv2d(h_1, 128, 32, 3, 2, tf.nn.leaky_relu, tflayers.layer_norm, regularizer, "h_2")  # 256 -> 128
    h_3 = tflayers.conv2d(h_2, 64, 3, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer,  scope="h_3")  # 128 -> 64
    h_4 = tflayers.conv2d(h_3, 128, 3, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer,  scope="h_4")  # 64 -> 32
    h_5 = tflayers.conv2d(unpool_2x2(h_4), 64, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer,  scope="h_5")  # 32 -> 64
    h_6 = tflayers.conv2d(unpool_2x2(h_5), 32, 3, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=tflayers.layer_norm, weights_regularizer=regularizer,  scope="h_6")  # 64 -> 128
    h_7 = block_conv2d(unpool_2x2(h_6, "unpool_7"), 128, 16, 3, 1, tf.nn.leaky_relu, tflayers.layer_norm, regularizer, "h_7")  # 128 -> 256
    h_8 = block_conv2d(unpool_2x2(h_7, "unpool_8"), 128, 5, 3, 1, None, None, regularizer, "h_8")  # 256 -> 512
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_8, labels=tf.one_hot(_y, 5))) + tf.contrib.layers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    predict = tf.cast(tf.argmax(h_8, axis=3), tf.uint8)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, _y), tf.float32) * tf.cast(tf.not_equal(_y, 0), tf.float32)) / tf.reduce_sum(tf.cast(tf.not_equal(_y, 0), tf.float32))
    return loss, accuracy, predict


def train(_batch_size, _block_size, step_start=0, restore=False):
    m_x, m_y = read_data("data/train_512_1/", _block_size)
    m_x, m_y = tf.train.shuffle_batch([m_x, m_y], batch_size=_batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(m_x, m_y, _batch_size, _block_size)
    tf.summary.scalar("loss", m_loss)
    merged_summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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
    batch_size = 4
    block_size = 512
    train(batch_size, block_size, 0, False)
