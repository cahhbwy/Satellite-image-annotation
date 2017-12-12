# coding:utf-8

from utils import *
import tensorflow as tf


def model(x, y, _batch_size, _image_size):
    h_0 = tf.divide(tf.cast(x, tf.float32), 255.)
    h_1 = conv2d(h_0, 8, 7, 7, [1, 2, 2, 1], "conv2d_1")
    h_2 = lrelu(batch_norm(conv2d(h_1, 8, 7, 7, [1, 2, 2, 1], "conv2d_2"), name="bn_2"))
    h_3 = conv2d(h_2, 16, 5, 5, [1, 1, 1, 1], "conv2d_3")
    h_4 = conv2d(h_3, 16, 5, 5, [1, 1, 1, 1], "conv2d_4")
    h_5 = lrelu(batch_norm(conv2d(h_4, 16, 5, 5, [1, 1, 1, 1], "conv2d_5"), name="bn_5"))
    h_6 = tf.image.resize_images(h_5, [_image_size, _image_size])
    h_7 = tf.nn.sigmoid(conv2d(h_6, 8, 5, 5, [1, 1, 1, 1], "conv2d_7"))
    h_8 = conv2d(h_7, 5, 5, 5, [1, 1, 1, 1], "conv2d_8")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_8, labels=tf.one_hot(y, 5)))
    predict = tf.cast(tf.argmax(h_8, axis=3), tf.uint8)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))
    return loss, accuracy, predict


if __name__ == '__main__':
    batch_size = 16
    image_size = 512
    X, Y = read_data(dir_path="data/train_512_1/", block_size=image_size)
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(X, Y, batch_size, image_size)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(m_loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    saver = tf.train.Saver(max_to_keep=10)

    for step in range(100000):
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: step})
        if step % 10 == 0:
            print("step %6d: loss = %f" % (step, v_loss))
        if step % 100 == 0:
            v_accuracy, origin, marking = sess.run([m_accuracy, X, m_predict])
            save_sample(origin, marking, "sample/sample_%05d.png" % step)
            print("step %6d: acc = %f" % (step, v_accuracy))
            saver.save(sess, "./models/model.ckpt", global_step=step)

