# coding:utf-8
# 卷积神经网络，拉普拉斯金字塔

from utils import *
import tensorflow as tf


def model(x, y, _batch_size, _image_size):
    h_0 = tf.divide(tf.cast(x, tf.float32), 255.)
    y_image = tf.reshape(y, (_batch_size, _image_size, _image_size, 1))

    y_16 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [_image_size // 16, _image_size // 16]), tf.uint8), [_batch_size, _image_size // 16, _image_size // 16])
    h0_0 = tf.image.resize_images(h_0, [_image_size // 16, _image_size // 16])
    h0_1 = tf.nn.sigmoid(conv2d(h0_0, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_0_1"))
    h0_2 = conv2d(h0_1, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="conv2d_0_2")
    loss_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h0_2, labels=tf.one_hot(y_16, 5)))

    y_08 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [_image_size // 8, _image_size // 8]), tf.uint8), [_batch_size, _image_size // 8, _image_size // 8])
    h1_0 = tf.image.resize_images(tf.nn.softmax(h0_2, dim=3), [_image_size // 8, _image_size // 8])
    h1_1 = tf.nn.sigmoid(conv2d(h1_0, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_1_1"))
    h1_2 = conv2d(h1_1, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="conv2d_1_2")
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h1_2, labels=tf.one_hot(y_08, 5)))

    y_04 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [_image_size // 4, _image_size // 4]), tf.uint8), [_batch_size, _image_size // 4, _image_size // 4])
    h2_0 = tf.image.resize_images(tf.nn.softmax(h1_2, dim=3), [_image_size // 4, _image_size // 4])
    h2_1 = lrelu(conv2d(h2_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="conv2d_2_1"))
    h2_2 = lrelu(conv2d(h2_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_2_2"))
    h2_3 = conv2d(h2_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="conv2d_2_3")
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2_3, labels=tf.one_hot(y_04, 5)))

    y_02 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [_image_size // 2, _image_size // 2]), tf.uint8), [_batch_size, _image_size // 2, _image_size // 2])
    h3_0 = tf.image.resize_images(tf.nn.softmax(h2_3, dim=3), [_image_size // 2, _image_size // 2])
    h3_1 = lrelu(conv2d(h3_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="conv2d_3_1"))
    h3_2 = lrelu(conv2d(h3_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_3_2"))
    h3_3 = conv2d(h3_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="conv2d_3_3")
    loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3_3, labels=tf.one_hot(y_02, 5)))

    y_01 = tf.reshape(y_image, [_batch_size, _image_size, _image_size])
    h4_0 = tf.image.resize_images(tf.nn.softmax(h3_3, dim=3), [_image_size, _image_size])
    h4_1 = lrelu(conv2d(h4_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="conv2d_4_1"))
    h4_2 = lrelu(conv2d(h4_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="conv2d_4_2"))
    h4_3 = conv2d(h4_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="conv2d_4_3")
    loss_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4_3, labels=tf.one_hot(y_01, 5)))

    _predict = tf.cast(tf.argmax(h4_3, axis=3), tf.uint8)
    _accuracy = tf.reduce_mean(tf.cast(tf.equal(_predict, y), tf.float32))

    return [loss_0, loss_1, loss_2, loss_3, loss_4], _accuracy, _predict


if __name__ == '__main__':
    batch_size = 4
    image_size = 512
    X, Y = read_data(dir_path="data/train_512_1/", block_size=image_size)
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(X, Y, batch_size, image_size)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=100, decay_rate=0.90)
    var_list = [[var for var in tf.trainable_variables() if "conv2d_" + str(i) in var.name] for i in range(5)]
    op = list()
    op.append(tf.train.AdamOptimizer(learning_rate).minimize(m_loss[0], var_list=var_list[0]))
    op.append(tf.train.AdamOptimizer(learning_rate).minimize(m_loss[1], var_list=var_list[1]))
    op.append(tf.train.AdamOptimizer(learning_rate).minimize(m_loss[2], var_list=var_list[2]))
    op.append(tf.train.AdamOptimizer(learning_rate).minimize(m_loss[3], var_list=var_list[3]))
    op.append(tf.train.AdamOptimizer(learning_rate).minimize(m_loss[4], var_list=var_list[4]))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    i = 0
    loss_queue = [10.0] * 100
    loss_mean = 10.0
    threshold = [0.08, 0.09, 0.10, 0.11, 0.12]
    for step in range(100000):
        _, v_loss = sess.run([op[i], m_loss[i]], feed_dict={global_step: step})
        loss_mean = (loss_mean * 100. - loss_queue[0] + v_loss) / 100.
        loss_queue[0:1] = []
        loss_queue.append(v_loss)
        if step % 10 == 0:
            loss_std = np.std(loss_queue)
            print("step %6d: loss = %f, mean of loss = %f, std of loss = %f" % (step, v_loss, loss_mean, loss_std))
            if loss_std < threshold[i] and i < 4:
                i += 1
                loss_queue = [10.0] * 100
                loss_mean = 10.0
        if i == 4 and step % 100 == 0:
            v_accuracy, origin, marking = sess.run([m_accuracy, X, m_predict])
            save_sample(origin, marking, "data/sample/sample_%05d.png" % step)
            print("step %6d: acc = %f" % (step, v_accuracy))
