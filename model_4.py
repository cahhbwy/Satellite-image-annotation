# coding:utf-8
# 卷积神经网络，拉普拉斯金字塔

from utils import *
import tensorflow as tf


def model(x, y, _batch_size, _image_size):
    s_01, s_02, s_04, s_08, s_16 = _image_size, _image_size // 2, _image_size // 4, _image_size // 8, _image_size // 16
    x_01 = tf.divide(tf.cast(x, tf.float32), 255.)
    y_image = tf.reshape(y, (-1, _image_size, _image_size, 1))

    x_16 = tf.image.resize_images(x_01, [s_16, s_16])
    y_16 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [s_16, s_16]), tf.uint8), [-1, s_16, s_16])
    h0_0 = x_16
    h0_1 = lrelu(conv2d(h0_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="L0_conv2d_1"))
    h0_2 = lrelu(conv2d(h0_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="L0_conv2d_2"))
    h0_3 = conv2d(h0_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="L0_conv2d_3")
    loss_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h0_3, labels=tf.one_hot(y_16, 5)))
    predict_0 = tf.cast(tf.argmax(h0_3, axis=3), tf.uint8)
    accuracy_0 = tf.reduce_mean(tf.cast(tf.equal(predict_0, y_16), tf.float32))
    x_0 = tf.cast(tf.multiply(x_16, 255.), tf.uint8)

    x_08 = tf.image.resize_images(x_01, [s_08, s_08])
    y_08 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [s_08, s_08]), tf.uint8), [-1, s_08, s_08])
    h1_0 = tf.concat([tf.image.resize_images(h0_3, [s_08, s_08]), x_08], axis=3, name="L1_concat_0")
    h1_1 = lrelu(conv2d(h1_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="L1_conv2d_1"))
    h1_2 = lrelu(conv2d(h1_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="L1_conv2d_2"))
    h1_3 = conv2d(h1_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="L1_conv2d_3")
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h1_3, labels=tf.one_hot(y_08, 5)))
    predict_1 = tf.cast(tf.argmax(h1_3, axis=3), tf.uint8)
    accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(predict_1, y_08), tf.float32))
    x_1 = tf.cast(tf.multiply(x_08, 255.), tf.uint8)

    x_04 = tf.image.resize_images(x_01, [s_04, s_04])
    y_04 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [s_04, s_04]), tf.uint8), [-1, s_04, s_04])
    h2_0 = tf.concat([tf.image.resize_images(h1_3, [s_04, s_04]), x_04], axis=3, name="L2_concat_0")
    h2_1 = lrelu(conv2d(h2_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="L2_conv2d_1"))
    h2_2 = lrelu(conv2d(h2_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="L2_conv2d_2"))
    h2_3 = conv2d(h2_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="L2_conv2d_3")
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2_3, labels=tf.one_hot(y_04, 5)))
    predict_2 = tf.cast(tf.argmax(h2_3, axis=3), tf.uint8)
    accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(predict_2, y_04), tf.float32))
    x_2 = tf.cast(tf.multiply(x_04, 255.), tf.uint8)

    x_02 = tf.image.resize_images(x_01, [s_02, s_02])
    y_02 = tf.reshape(tf.cast(tf.image.resize_images(y_image, [s_02, s_02]), tf.uint8), [-1, s_02, s_02])
    h3_0 = tf.concat([tf.image.resize_images(h2_3, [s_02, s_02]), x_02], axis=3, name="L3_concat_0")
    h3_1 = lrelu(conv2d(h3_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="L3_conv2d_1"))
    h3_2 = lrelu(conv2d(h3_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="L3_conv2d_2"))
    h3_3 = conv2d(h3_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="L3_conv2d_3")
    loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h1_3, labels=tf.one_hot(y_02, 5)))
    predict_3 = tf.cast(tf.argmax(h3_3, axis=3), tf.uint8)
    accuracy_3 = tf.reduce_mean(tf.cast(tf.equal(predict_3, y_02), tf.float32))
    x_3 = tf.cast(tf.multiply(x_02, 255.), tf.uint8)

    h4_0 = tf.concat([tf.image.resize_images(h3_3, [s_01, s_01]), x_01], axis=3, name="L4_concat_0")
    h4_1 = lrelu(conv2d(h4_0, 10, k_h=11, k_w=11, strides=(1, 1, 1, 1), name="L4_conv2d_1"))
    h4_2 = lrelu(conv2d(h4_1, 20, k_h=7, k_w=7, strides=(1, 1, 1, 1), name="L4_conv2d_2"))
    h4_3 = conv2d(h4_2, 5, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="L4_conv2d_3")
    loss_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4_3, labels=tf.one_hot(y, 5)))
    predict_4 = tf.cast(tf.argmax(h4_3, axis=3), tf.uint8)
    accuracy_4 = tf.reduce_mean(tf.cast(tf.equal(predict_4, y), tf.float32))
    x_4 = x

    return [loss_0, loss_1, loss_2, loss_3, loss_4], [accuracy_0, accuracy_1, accuracy_2, accuracy_3, accuracy_4], \
           [predict_0, predict_1, predict_2, predict_3, predict_4], [x_0, x_1, x_2, x_3, x_4]


if __name__ == '__main__':
    batch_size = tf.Variable(0, trainable=False)
    X, Y = tf.train.shuffle_batch(read_data("data/train_512_1/", 512), batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict, m_x = model(X, Y, batch_size, 512)
    var_list = [[var for var in tf.trainable_variables() if name in var.name] for name in ["L0", "L1", "L2", "L3", "L4"]]
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=100, decay_rate=0.95)
    op = [tf.train.AdamOptimizer(learning_rate).minimize(m_loss[i], var_list=[var for j in range(i + 1) for var in var_list[j]]) for i in range(5)]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, "./models/model.ckpt-18000")

    index = 1
    loss_queue = np.random.randint(0, 100, 100)
    threshold = [0.08, 0.09, 0.10, 0.11, 0.12]
    batch_size_list = [256, 144, 64, 36, 9]
    for step in range(18001, 100000):
        _, v_loss = sess.run([op[index], m_loss[index]], feed_dict={global_step: step, batch_size: batch_size_list[index]})
        loss_queue = np.delete(loss_queue, 0)
        loss_queue = np.append(loss_queue, v_loss)
        if step % 10 == 0:
            loss_mean = np.mean(loss_queue)
            loss_std = np.std(loss_queue)
            print("level %d, step %6d: loss = %f, mean of loss = %f, std of loss = %f, std / mean = %f" % (index, step, v_loss, loss_mean, loss_std, loss_std / loss_mean))
            if index < 4 and loss_std < 0.5 and loss_std / loss_mean < 0.05:
                index += 1
                loss_queue = np.random.randint(0, 100, 100)
        if step % 100 == 0:
            v_accuracy, origin, marking = sess.run([m_accuracy[index], m_x[index], m_predict[index]], feed_dict={batch_size: batch_size_list[index]})
            save_sample(origin, marking, "data/sample/sample_%05d.png" % step)
            print("step %6d: acc = %f" % (step, v_accuracy))
            saver.save(sess, "./models/model.ckpt", global_step=step)
