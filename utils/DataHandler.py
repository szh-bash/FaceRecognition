import cv2
import numpy as np
from config import batch_size


_CH = 32
_CW = 32
W = 112
H = 112
MinS = 112
# MaxS = 128
MaxS = 112


def mixup(rng, image, label, image_sec, label_sec, alpha=0.05):
    lam = rng.beta(alpha, alpha)
    # print(lam)  # same in diff gpu
    img = image * lam + image_sec * (1-lam)
    lb = label * lam + label_sec * (1-lam)
    return img, lb


def cutout(rng, img):
    img = img.copy()
    if rng.rand() < 0.1:
        y = int(rng.rand() * H)
        x = int(rng.rand() * W)
        img[y:min(H, y+_CH), x:min(W, x+_CW), :] = 0
    return img


def resize(rng, img):
    img = img.copy()
    new_size = rng.randint(MinS, MaxS+1)
    img = cv2.resize(img, (new_size, new_size))
    return img


def crop(rng, img):
    img = img.copy()
    dh = img.shape[1] - H
    dw = img.shape[0] - W
    y = int(rng.rand()*dh)
    x = int(rng.rand()*dw)
    return img[y:y+H, x:x+W, :]


def flip(rng, img):
    img = img.copy()
    if rng.rand() < 0.5:
        img = img[:, ::-1, :]
    return img


def gaussian_blur(rng, img):
    img = img.copy()
    if rng.rand() < 0.3:
        img = cv2.blur(img, (5, 5))
    return img


def rotate(rng, img):
    img = img.copy()
    if rng.rand() < 0.3:
        ang = rng.rand()*30-15
        mat = cv2.getRotationMatrix2D((H / 2, W / 2), ang, 1)
        img = cv2.warpAffine(img, mat, (H, W), borderValue=[0, 0, 0])
    return img


def trans(rng, img):
    img = img.copy()
    if rng.rand() < 0.3:
        dh = (rng.rand()*H - H/2)*0.2
        dw = (rng.rand()*W - W/2)*0.2
        mat = np.array([[1, 0, dh], [0, 1, dw]], dtype=np.float64)
        img = cv2.warpAffine(img, mat, (H, W), borderValue=[0, 0, 0])
    return img


def run(rng, img):
    img = cutout(rng, img)
    # img = rotate(rng, img)
    # img = trans(rng, img)
    img = gaussian_blur(rng, img)
    img = flip(rng, img)
    img = np.transpose(img, [2, 0, 1])
    img = (img - 127.5) / 128.0
    return img


# def run2(rng, image, label, image_sec, label_sec):
#     img = run(rng, image)
#     img2 = run(rng, image_sec)
#     img, lb = mixup(rng, img, label, img2, label_sec)
#     return img, lb
