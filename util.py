# coding:utf-8

from PIL import Image
import numpy as np


# 0. 其他 (  0,   0,   0)
# 1. 植被 (  0 ,255,   0)
# 2. 道路 (255 ,255,   0)
# 3. 建筑 (128 ,128, 128)
# 4. 水体 (  0 ,  0, 255)
def visualization(origin, mask, alpha=0.5):
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    colors = [[0, 0, 0], [0, 255, 0], [255, 255, 0], [128, 128, 128], [0, 0, 255]]
    img = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        x, y = np.where(mask == i)
        img[x, y, :] = colors[i]
    return Image.fromarray((alpha * img + (1 - alpha) * origin).astype(np.uint8))


def make_ndarray(filename_images, filename_masks=None, sample_size=(512, 512), amount=1000, scale=4):
    if filename_masks is not None:
        assert len(filename_images) == len(filename_masks)
    images = [Image.open(fn) for fn in filename_images]
    images = [image.resize((image.size[0] // scale, image.size[1] // scale), Image.BILINEAR) for image in images]
    images = [np.array(image) for image in images]
    if filename_masks is not None:
        masks = [Image.open(fn) for fn in filename_masks]
        masks = [mask.resize((mask.size[0] // scale, mask.size[1] // scale), Image.NEAREST) for mask in masks]
        masks = [np.array(mask) for mask in masks]
    else:
        masks = []
    amounts = np.sort(np.random.choice(amount, len(images) - 1))
    amounts = [amounts[0]] + [i2 - i1 for i1, i2 in zip(amounts[:-1], amounts[1:])] + [amount - amounts[-1]]
    data_images = []
    data_masks = []
    for idx in range(len(images)):
        pos_u = np.random.randint(0, images[idx].shape[0] - sample_size[0], amounts[idx])
        pos_d = pos_u + sample_size[0]
        pos_l = np.random.randint(0, images[idx].shape[1] - sample_size[1], amounts[idx])
        pos_r = pos_l + sample_size[1]
        data_images.append(np.array([images[idx][pos_u[i]:pos_d[i], pos_l[i]:pos_r[i], :] for i in range(amounts[idx])]))
        if filename_masks is not None:
            data_masks.append(np.array([masks[idx][pos_u[i]:pos_d[i], pos_l[i]:pos_r[i]] for i in range(amounts[idx])]))
    data_images = [data_image for data_image in data_images if len(data_image) > 0]
    if filename_masks is None:
        return np.concatenate(data_images, axis=0)
    else:
        data_masks = [data_mask for data_mask in data_masks if len(data_mask) > 0]
        return np.concatenate(data_images, axis=0), np.concatenate(data_masks, axis=0)


def calc_weight(masks, stride=1, base=1.0):
    weights = np.ones(masks.shape) * base
    for x in range(masks.shape[1]):
        for y in range(masks.shape[1]):
            x_p = max(0, x - stride)
            x_n = min(masks.shape[1], x + stride + 1)
            y_p = max(0, y - stride)
            y_n = min(masks.shape[2], y + stride + 1)
            weights[:, x, y] += 1 - np.mean(masks[:, x_p:x_n, y_p:y_n] == masks[:, x: x + 1, y: y + 1], axis=(1, 2))
    weights = weights / weights.sum(axis=(1, 2), keepdims=True) * (masks.shape[1] * masks.shape[2])
    return weights.astype(np.float32)


if __name__ == '__main__':
    # origin = Image.open("data/train_1_8bits.png")
    # mask = Image.open("data/train_1_class_8bits.png")
    # img = visualization(origin, mask)
    # img.save("sample/tmp.png")

    # images, masks = make_ndarray(
    #     ["data/train_1_8bits.png", "data/train_2_8bits.png"],
    #     ["data/train_1_class_8bits.png", "data/train_2_class_8bits.png"],
    #     (512, 512), 1000, 4)
    # np.savez("data/data.npz", images=images, masks=masks)

    masks = np.random.randint(0, 3, size=(2, 5, 5))
    print(masks)
    weights = calc_weight(masks)
    print(weights)
    pass
