import cv2
import numpy as np
import os
import time
import torchvision.transforms as trans
from config import batch_size


_CH = 32
_CW = 32
W = 112
H = 112
MinS = 112
MaxS = 128


def mixup(rng, image, label, image_sec, label_sec, alpha=0.05):
    lam = rng.beta(alpha, alpha)
    # print(lam)  # same in diff gpu
    img = image * lam + image_sec * (1-lam)
    lb = label * lam + label_sec * (1-lam)
    return img, lb


def cutout(rng, image):
    img = image.copy()
    if rng.rand() < 0.5:
        y = int(rng.rand() * H)
        x = int(rng.rand() * W)
        img[y:min(H, y+_CH), x:min(W, x+_CW), :] = 0
    return img


def resize(rng, image):
    img = image.copy()
    # print(rng.rand())
    new_size = rng.randint(MinS, MaxS+1)
    img = cv2.resize(img, (new_size, new_size))
    return img


def crop(rng, image):
    img = image.copy()
    dh = img.shape[1] - H
    dw = img.shape[0] - W
    # print(rng.rand())
    y = int(rng.rand()*dh)
    x = int(rng.rand()*dw)
    return img[y:y+H, x:x+W, :]


def rotate(rng, image):
    img = image.copy()
    if rng.rand() < 0.3:
        img = trans.RandomRotation(img, 15)  # -15 -> +15
    return img


def flip(rng, image):
    img = image.copy()
    # print(rng.rand())
    if rng.rand() < 0.5:
        img = img[:, ::-1, :]
    return img


def gaussian_blur(rng, image):
    img = image.copy()
    # print(rng.rand())
    if rng.rand() < 0.3:
        img = cv2.blur(img, (5, 5))
    return img


def run(rng, image):
    img = image.copy()
    img = cv2.resize(img, (H, W))
    img = resize(rng, img)
    img = gaussian_blur(rng, img)
    img = crop(rng, img)
    img = flip(rng, img)
    img = np.transpose(img, [2, 0, 1])
    img = (img - 127.5) / 128.0
    return img


def run2(image, label, image_sec, label_sec):
    img = run(image)
    img2 = run(image_sec)
    img, lb = mixup(img, label, img2, label_sec)
    return img, lb
