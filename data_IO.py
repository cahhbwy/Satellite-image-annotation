# coding:utf-8

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import multiprocessing


# 0. 其他 (  0,   0,   0)
# 1. 植被 (  0 ,255,   0)
# 2. 道路 (255 ,255,   0)
# 3. 建筑 (128 ,128, 128)
# 4. 水体 (  0 ,  0, 255)
def visualization(_origin, _marking, alpha=0.5, show=True, save=False, _save_path="temp.png"):
    if not isinstance(_origin, np.ndarray):
        _origin = np.array(_origin)
    if not isinstance(_marking, np.ndarray):
        _marking = np.array(_marking)
    colors = [[0, 0, 0], [0, 255, 0], [255, 255, 0], [128, 128, 128], [0, 0, 255]]
    img = np.zeros(shape=(_marking.shape[0], _marking.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        x, y = np.where(_marking == i)
        img[x, y, :] = colors[i]
    if save:
        Image.fromarray((alpha * img + (1 - alpha) * _origin).astype(np.uint8)).save(_save_path)
    if show:
        Image.fromarray((alpha * img + (1 - alpha) * _origin).astype(np.uint8)).show()


def create_tfrecord(_origin, _marking, _filename, _block_size, _data_block):
    assert _origin.size == _marking.size
    print("create %s..." % _filename)
    writer = tf.python_io.TFRecordWriter(_filename)
    x = np.random.randint(0, _origin.size[0] - _block_size, _data_block)
    y = np.random.randint(0, _origin.size[1] - _block_size, _data_block)
    for i in range(_data_block):
        _origin_bytes = _origin.crop((x[i], y[i], x[i] + _block_size, y[i] + _block_size)).tobytes()
        _marking_bytes = _marking.crop((x[i], y[i], x[i] + _block_size, y[i] + _block_size)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[_origin_bytes])),
            'marking': tf.train.Feature(bytes_list=tf.train.BytesList(value=[_marking_bytes]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("create %s finished." % _filename)


def create_tfrecords_multi_threads(_origins, _markings, _save_path="data/train", _block_size=128, _total=65536, _file_size=256, limit_threads_num=None):
    assert len(_origins) == len(_markings)
    if limit_threads_num is None:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(limit_threads_num)
    data_block = (_file_size * 1024 * 1024) // (_block_size * _block_size * 4)
    params = []
    for i in range(_total // data_block):
        choice = np.random.randint(0, 2)
        params.append((_origins[choice], _markings[choice], os.path.join(_save_path, "%04d.tfrecords" % i), _block_size, data_block))
    pool.starmap(create_tfrecord, params)


def read_data(dir_path="data/train/", _block_size=256):
    file_list = [dir_path + filename for filename in os.listdir(dir_path)]
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example, features={
        'origin': tf.FixedLenFeature([], tf.string),
        'marking': tf.FixedLenFeature([], tf.string)
    })
    _origin = tf.decode_raw(features['origin'], tf.uint8)
    _marking = tf.decode_raw(features['marking'], tf.uint8)
    _origin = tf.reshape(_origin, (_block_size, _block_size, 3))
    _marking = tf.reshape(_marking, (_block_size, _block_size))
    return _origin, _marking


def save_sample(origin_ndarry, marking_ndarry, path="temp.png"):
    count = origin_ndarry.shape[0]
    size = origin_ndarry.shape[1]
    row = np.ceil(np.sqrt(count)).astype(np.int32)
    col = count // row
    img_h = row * size
    img_w = col * size
    _origin = np.zeros(shape=[img_h, img_w, 3], dtype=np.uint8)
    _marking = np.ones(shape=[img_h, img_w], dtype=np.uint8) * 255
    for i in range(count):
        x = (i // col) * size
        y = (i % col) * size
        _origin[x:x + size, y:y + size, :] = origin_ndarry[i]
        _marking[x:x + size, y:y + size] = marking_ndarry[i]
    visualization(_origin, _marking, show=False, save=True, _save_path=path)


def image_resize_by_mode(_image, _scale):
    img = np.array(_image)
    if len(img.shape) == 2:
        img = img.reshape([img.shape[0], img.shape[1], 1])
    shape = img.shape
    out = np.zeros((shape[0] // _scale, shape[1] // _scale, shape[2]), np.uint8)
    x = np.arange((shape[0] % _scale) // 2, shape[0], _scale)
    y = np.arange((shape[1] % _scale) // 2, shape[1], _scale)
    for c in range(shape[2]):
        for i in range(x.size - 1):
            for j in range(y.size - 1):
                tmp = img[x[i]:x[i + 1], y[j]:y[j + 1], c].reshape(_scale * _scale)
                out[i, j, c] = np.argmax(np.bincount(tmp))
    if out.shape[2] == 1:
        out = out.reshape((shape[0] // _scale, shape[1] // _scale))
    return Image.fromarray(out)


def unpool_2x2(x, scope="unpool"):
    new_height = x.get_shape()[1].value * 2
    new_width = x.get_shape()[2].value * 2
    with tf.variable_scope(scope):
        return tf.image.resize_nearest_neighbor(x, (new_height, new_width))


if __name__ == '__main__':
    save_path = "data/train_128_4"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_size = 512
    block_size = 128
    total = 262144
    scale = 4
    origins = [Image.open("data/CCF-training/1-8bits.png"), Image.open("data/CCF-training/2-8bits.png")]
    markings = [Image.open("data/CCF-training/1_class_8bits.png"), Image.open("data/CCF-training/2_class_8bits.png")]
    if scale > 1:
        origins[0] = origins[0].resize((origins[0].size[0] // scale, origins[0].size[1] // scale), Image.BILINEAR)
        origins[1] = origins[1].resize((origins[1].size[0] // scale, origins[1].size[1] // scale), Image.BILINEAR)
        markings[0] = image_resize_by_mode(markings[0], scale)
        markings[1] = image_resize_by_mode(markings[1], scale)
    create_tfrecords_multi_threads(origins, markings, save_path, block_size, total, file_size)
    # visualization(Image.open("data/CCF-training/1-8bits.png"), Image.open("data/CCF-training/1_class_8bits.png"), save=True)
