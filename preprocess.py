# coding:utf-8

from PIL import Image
import numpy as np
import tensorflow as tf


# 0. 其他 ( 64 , 64,  64)
# 1. 植被 (  0 ,128,   0)
# 2. 道路 (255 ,255,   0)
# 3. 建筑 (128 ,128, 128)
# 4. 水体 (  0 ,  0, 255)
def visualization(origin, marking, alpha=0.5, show=True, save=False):
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    if not isinstance(marking, np.ndarray):
        marking = np.array(marking)
    colors = [[64, 64, 64], [0, 128, 0], [255, 255, 0], [128, 128, 128], [0, 0, 255]]
    img = np.zeros(shape=(marking.shape[0], marking.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        x, y = np.where(marking == i)
        img[x, y, :] = colors[i]
    if save:
        Image.fromarray(img).save("temp_1.png")
        Image.fromarray((alpha * img + (1 - alpha) * origin).astype(np.uint8)).save("temp_2.png")
    if show:
        Image.fromarray((alpha * img + (1 - alpha) * origin).astype(np.uint8)).show()


def split_data(origin, marking, block_size=256, overlapping=0.3, use_tf=False, tfrecords_size=1024, random=False, total=8192):
    shape = marking.size
    if random:
        x = np.random.randint(0, shape[0] - block_size, total)
        y = np.random.randint(0, shape[1] - block_size, total)
        if use_tf:
            count = 0
            file_num = 0
            for i in range(total):
                if count == 0:
                    print("create data_%04d.tfrecords" % file_num)
                    writer = tf.python_io.TFRecordWriter("data/train/data_%04d.tfrecords" % file_num)
                origin_raw = origin.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).tobytes()
                marking_raw = marking.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[origin_raw])),
                    'marking': tf.train.Feature(bytes_list=tf.train.BytesList(value=[marking_raw]))
                }))
                writer.write(example.SerializeToString())
                count += 1
                if count >= tfrecords_size:
                    count = 0
                    file_num += 1
                    writer.close()
        else:
            for i in range(total):
                origin.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).save("data/train/%06d.png" % i)
                marking.crop((x[i], y[i], x[i] + block_size, y[i] + block_size)).save("data/train/%06d_class.png" % i)
    else:
        i = 0
        if use_tf:
            count = 0
            file_num = 0
            for x in range(0, shape[0] - block_size, int(block_size * (1 - overlapping))):
                for y in range(0, shape[1] - block_size, int(block_size * (1 - overlapping))):
                    if count == 0:
                        print("create data_%04d.tfrecords" % file_num)
                        writer = tf.python_io.TFRecordWriter("data/train/data_%04d.tfrecords" % file_num)
                    origin_raw = origin.crop((x, y, x + block_size, y + block_size)).tobytes()
                    marking_raw = marking.crop((x, y, x + block_size, y + block_size)).tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[origin_raw])),
                        'marking': tf.train.Feature(bytes_list=tf.train.BytesList(value=[marking_raw]))
                    }))
                    writer.write(example.SerializeToString())
                    count += 1
                    if count >= tfrecords_size:
                        count = 0
                        file_num += 1
                        writer.close()
        else:
            for x in range(0, shape[0] - block_size, int(block_size * (1 - overlapping))):
                for y in range(0, shape[1] - block_size, int(block_size * (1 - overlapping))):
                    origin.crop((x, y, x + block_size, y + block_size)).save("data/train/%06d.png" % i)
                    marking.crop((x, y, x + block_size, y + block_size)).save("data/train/%06d_class.png" % i)
                    i += 1


if __name__ == '__main__':
    _marking = Image.open("data/CCF-training/1_class_8bits.png")
    _origin = Image.open("data/CCF-training/1-8bits.png")

    # visualization(_origin, _marking, 0.5)
    split_data(_origin, _marking, use_tf=True, random=True)
