# coding:utf-8

from utils import *
import tensorflow as tf


def model(x, y, _batch_size, _image_size):
    h_0 = tf.divide(tf.cast(x, tf.float32), 255.)
    h_11 = lrelu(batch_norm(conv2d(h_0, 8, 3, 3, [1, 1, 1, 1], "conv2d_11"), name="bn_11"))
    h_12 = lrelu(batch_norm(conv2d(h_11, 16, 3, 3, [1, 1, 1, 1], "conv2d_12"), name="bn_12"))
    h_13 = max_pool_2x2(h_12, "pool_1")

    h_21 = lrelu(batch_norm(conv2d(h_13, 16, 3, 3, [1, 1, 1, 1], "conv2d_21"), name="bn_21"))
    h_22 = lrelu(batch_norm(conv2d(h_21, 32, 3, 3, [1, 1, 1, 1], "conv2d_22"), name="bn_22"))
    h_23 = max_pool_2x2(h_22, "pool_2")

    h_31 = lrelu(batch_norm(conv2d(h_23, 32, 3, 3, [1, 1, 1, 1], "conv2d_31"), name="bn_31"))
    h_32 = lrelu(batch_norm(conv2d(h_31, 32, 3, 3, [1, 1, 1, 1], "conv2d_32"), name="bn_32"))
    h_33 = lrelu(batch_norm(conv2d(h_32, 64, 3, 3, [1, 1, 1, 1], "conv2d_33"), name="bn_33"))
    h_34 = max_pool_2x2(h_33, "pool_3")

    h_41 = lrelu(batch_norm(conv2d(h_34, 64, 3, 3, [1, 1, 1, 1], "conv2d_41"), name="bn_41"))
    h_42 = lrelu(batch_norm(conv2d(h_41, 64, 3, 3, [1, 1, 1, 1], "conv2d_42"), name="bn_42"))
    h_43 = lrelu(batch_norm(conv2d(h_42, 128, 3, 3, [1, 1, 1, 1], "conv2d_43"), name="bn_43"))
    h_44 = max_pool_2x2(h_43, "pool_4")

    h_51 = lrelu(batch_norm(conv2d(h_44, 128, 3, 3, [1, 1, 1, 1], "conv2d_51"), name="bn_51"))
    h_52 = lrelu(batch_norm(conv2d(h_51, 128, 3, 3, [1, 1, 1, 1], "conv2d_52"), name="bn_52"))
    h_53 = lrelu(batch_norm(conv2d(h_52, 256, 3, 3, [1, 1, 1, 1], "conv2d_53"), name="bn_53"))
    h_54 = max_pool_2x2(h_53, "pool_5")

    h_61 = max_unpool_2x2(h_54, "unpool_6")
    h_62 = lrelu(batch_norm(conv2d(h_61, 128, 3, 3, [1, 1, 1, 1], "conv2d_61"), name="bn_61"))
    h_63 = lrelu(batch_norm(conv2d(h_62, 128, 3, 3, [1, 1, 1, 1], "conv2d_62"), name="bn_62"))
    h_64 = lrelu(batch_norm(conv2d(h_63, 128, 3, 3, [1, 1, 1, 1], "conv2d_63"), name="bn_63"))

    h_71 = max_unpool_2x2(h_64, "unpool_7")
    h_72 = lrelu(batch_norm(conv2d(h_71, 64, 3, 3, [1, 1, 1, 1], "conv2d_71"), name="bn_71"))
    h_73 = lrelu(batch_norm(conv2d(h_72, 64, 3, 3, [1, 1, 1, 1], "conv2d_72"), name="bn_72"))
    h_74 = lrelu(batch_norm(conv2d(h_73, 64, 3, 3, [1, 1, 1, 1], "conv2d_73"), name="bn_73"))

    h_81 = max_unpool_2x2(h_74, "unpool_8")
    h_82 = lrelu(batch_norm(conv2d(h_81, 32, 3, 3, [1, 1, 1, 1], "conv2d_81"), name="bn_81"))
    h_83 = lrelu(batch_norm(conv2d(h_82, 32, 3, 3, [1, 1, 1, 1], "conv2d_82"), name="bn_82"))
    h_84 = lrelu(batch_norm(conv2d(h_83, 32, 3, 3, [1, 1, 1, 1], "conv2d_83"), name="bn_83"))

    h_91 = max_unpool_2x2(h_84, "unpool_9")
    h_92 = lrelu(batch_norm(conv2d(h_91, 16, 3, 3, [1, 1, 1, 1], "conv2d_91"), name="bn_91"))
    h_93 = lrelu(batch_norm(conv2d(h_92, 16, 3, 3, [1, 1, 1, 1], "conv2d_92"), name="bn_92"))

    h_101 = max_unpool_2x2(h_93, "unpool_9")
    h_102 = lrelu(batch_norm(conv2d(h_101, 8, 3, 3, [1, 1, 1, 1], "conv2d_101"), name="bn_101"))
    h_103 = lrelu(batch_norm(conv2d(h_102, 5, 3, 3, [1, 1, 1, 1], "conv2d_102"), name="bn_102"))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_103, labels=tf.one_hot(y, 5)))
    predict = tf.cast(tf.argmax(h_103, axis=3), tf.uint8)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))
    return loss, accuracy, predict


if __name__ == '__main__':
    batch_size = 16
    image_size = 256
    X, Y = read_data(dir_path="data/train_256_2/", block_size=image_size)
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(X, Y, batch_size, image_size)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
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
