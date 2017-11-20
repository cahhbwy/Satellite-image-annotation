# coding:utf-8
# 单独提取每一种，再做整合

from utils import *
import tensorflow as tf


# 0. 其他 ( 64 , 64,  64)
# 1. 植被 (  0 ,128,   0)
# 2. 道路 (255 ,255,   0)
# 3. 建筑 (128 ,128, 128)
# 4. 水体 (  0 ,  0, 255)
def model(x, y, _batch_size, _image_size):
    h_0 = tf.divide(tf.cast(x, tf.float32), 255.)
    # 0. 其他
    h0_0 = lrelu(conv2d(h_0, 50, k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h0_conv2d_0"))
    h0_1 = lrelu(conv2d(h0_0, 20, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h0_conv2d_1"))
    h0_2 = lrelu(conv2d(h0_1, 10, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h0_conv2d_2"))
    h0_3 = deconv2d(h0_2, (_batch_size, _image_size, _image_size, 1), k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h0_deconv2d_3")
    y_0 = tf.cast(tf.reshape(tf.equal(y, 1), (_batch_size, _image_size, _image_size, 1)), dtype=tf.float32)
    loss_0 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_0, logits=h0_3, name="loss_0"))
    # 1. 植被
    h1_0 = lrelu(conv2d(h_0, 50, k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h1_conv2d_0"))
    h1_1 = lrelu(conv2d(h1_0, 20, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h1_conv2d_1"))
    h1_2 = lrelu(conv2d(h1_1, 10, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h1_conv2d_2"))
    h1_3 = deconv2d(h1_2, (_batch_size, _image_size, _image_size, 1), k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h1_deconv2d_3")
    y_1 = tf.cast(tf.reshape(tf.equal(y, 1), (_batch_size, _image_size, _image_size, 1)), dtype=tf.float32)
    loss_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_1, logits=h1_3, name="loss_1"))
    # 2. 道路
    h2_0 = lrelu(conv2d(h_0, 50, k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h2_conv2d_0"))
    h2_1 = lrelu(conv2d(h2_0, 20, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h2_conv2d_1"))
    h2_2 = lrelu(conv2d(h2_1, 10, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h2_conv2d_2"))
    h2_3 = deconv2d(h2_2, (_batch_size, _image_size, _image_size, 1), k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h2_deconv2d_3")
    y_2 = tf.cast(tf.reshape(tf.equal(y, 1), (_batch_size, _image_size, _image_size, 1)), dtype=tf.float32)
    loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_2, logits=h2_3, name="loss_2"))
    # 3. 建筑
    h3_0 = lrelu(conv2d(h_0, 50, k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h3_conv2d_0"))
    h3_1 = lrelu(conv2d(h3_0, 20, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h3_conv2d_1"))
    h3_2 = lrelu(conv2d(h3_1, 10, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h3_conv2d_2"))
    h3_3 = deconv2d(h3_2, (_batch_size, _image_size, _image_size, 1), k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h3_deconv2d_3")
    y_3 = tf.cast(tf.reshape(tf.equal(y, 1), (_batch_size, _image_size, _image_size, 1)), dtype=tf.float32)
    loss_3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_3, logits=h3_3, name="loss_3"))
    # 4. 水体
    h4_0 = lrelu(conv2d(h_0, 50, k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h4_conv2d_0"))
    h4_1 = lrelu(conv2d(h4_0, 20, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h4_conv2d_1"))
    h4_2 = lrelu(conv2d(h4_1, 10, k_h=5, k_w=5, strides=(1, 1, 1, 1), name="h4_conv2d_2"))
    h4_3 = deconv2d(h4_2, (_batch_size, _image_size, _image_size, 1), k_h=9, k_w=9, strides=(1, 2, 2, 1), name="h4_deconv2d_3")
    y_4 = tf.cast(tf.reshape(tf.equal(y, 1), (_batch_size, _image_size, _image_size, 1)), dtype=tf.float32)
    loss_4 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_4, logits=h4_3, name="loss_3"))
    h_4 = tf.concat([h0_3, h1_3, h2_3, h3_3, h4_3], axis=3, name="concat_4")
    loss_total = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 5), logits=h_4)
    _loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_total
    _predict = tf.cast(tf.argmax(h_4, axis=3), tf.uint8)
    _accuracy = tf.reduce_mean(tf.cast(tf.equal(_predict, y), tf.float32))
    return _loss, _accuracy, _predict


if __name__ == '__main__':
    batch_size = 4
    image_size = 512
    X, Y = read_data(block_size=image_size)
    X, Y = tf.train.shuffle_batch([X, Y], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_accuracy, m_predict = model(X, Y, batch_size, image_size)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    for step in range(100000):
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: step})
        if step % 10 == 0:
            print("step %6d: loss = %f" % (step, v_loss))
        if step % 100 == 0:
            v_accuracy, origin, marking = sess.run([m_accuracy, X, m_predict])
            save_sample(origin, marking, "data/sample/sample_%05d.png" % step)
            print("step %6d: acc = %f" % (step, v_accuracy))
