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


def make_ndarray(crop_size, amount):
    images = [np.array(Image.open("data/train_1_8bits.png")), np.array(Image.open("data/train_2_8bits.png"))]
    masks = [np.array(Image.open("data/train_1_class_8bits.png")), np.array(Image.open("data/train_2_class_8bits.png"))]
    amount_1 = int(amount * np.random.random())
    amount_2 = amount - amount_1
    amount = [amount_1, amount_2]
    data_images = [None, None]
    data_masks = [None, None]
    for idx in [0, 1]:
        pos_u = np.random.randint(0, images[idx].shape[0] - crop_size[0], amount[idx])
        pos_d = pos_u + crop_size[0]
        pos_l = np.random.randint(0, images[idx].shape[1] - crop_size[1], amount[idx])
        pos_r = pos_l + crop_size[1]
        data_images[idx] = np.array([images[idx][pos_u[i]:pos_d[i], pos_l[i]:pos_r[i], :] for i in range(amount[idx])])
        data_masks[idx] = np.array([masks[idx][pos_u[i]:pos_d[i], pos_l[i]:pos_r[i]] for i in range(amount[idx])])
    return np.concatenate(data_images, axis=0), np.concatenate(data_masks, axis=0)


if __name__ == '__main__':
    # origin = Image.open("data/train_1_8bits.png")
    # mask = Image.open("data/train_1_class_8bits.png")
    # img = visualization(origin, mask)
    # img.save("sample/tmp.png")

    # images, masks = make_ndarray((512, 512), 1000)
    # np.savez("data/data.npz", images=images, masks=masks)

    pass
