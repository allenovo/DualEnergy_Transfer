import os
import math
import numpy as np


def montage(imgs, shape):
    assert len(imgs) == np.prod(shape)
    w, h = imgs[0].shape[0], imgs[0].shape[1]
    montage_img = np.zeros((h*shape[0], w*shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            img = imgs[i*shape[1]+j]
            montage_img[i*h:(i+1)*h,j*w:(j+1)*w] = img

    return montage_img


def normalize(imgs):
    # TODO!
    pass


def denormalize(imgs):
    # TODO!
    pass


