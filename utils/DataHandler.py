import cv2
import numpy as np
# import torch
import torchvision.transforms as trans
from config import batch_size


_CH = 32
_CW = 32
W = 112
H = 112
MinS = 112
MaxS = 128


class Augment:
    rng = np.random
    rng2 = np.random

    def mixup(self, image, label, image_sec, label_sec, alpha=0.05):
        lam = self.rng2.beta(alpha, alpha)
        # print(lam)  # same in diff gpu
        img = image * lam + image_sec * (1-lam)
        lb = label * lam + label_sec * (1-lam)
        return img, lb

    def cutout(self, image):
        img = image.copy()
        if self.rng.rand() < 0.5:
            y = int(self.rng.rand() * H)
            x = int(self.rng.rand() * W)
            img[y:min(H, y+_CH), x:min(W, x+_CW), :] = 0
        return img

    def resize(self, image):
        img = image.copy()
        new_size = self.rng.randint(MinS, MaxS+1)
        img = cv2.resize(img, (new_size, new_size))
        return img

    def crop(self, image):
        img = image.copy()
        dh = img.shape[1] - H
        dw = img.shape[0] - W
        y = int(self.rng.rand()*dh)
        x = int(self.rng.rand()*dw)
        return img[y:y+H, x:x+W, :]

    def rotate(self, image):
        img = image.copy()
        if self.rng.rand() < 0.3:
            img = trans.RandomRotation(img, 15)  # -15 -> +15
        return img

    def flip(self, image):
        img = image.copy()
        if self.rng.rand() < 0.5:
            img = img[:, ::-1, :]
        return img

    def gaussian_blur(self, image):
        img = image.copy()
        if self.rng.rand() < 0.3:
            img = cv2.blur(img, (5, 5))
        return img

    def run(self, image):
        img = image.copy()
        img = cv2.resize(img, (H, W))
        # img = self.cutout(img)
        img = self.resize(img)
        # img = self.rotate(img)
        img = self.gaussian_blur(img)
        img = self.crop(img)
        img = self.flip(img)
        img = np.transpose(img, [2, 0, 1])
        img = (img - 127.5) / 128.0

        return img

    def run2(self, image, label, image_sec, label_sec):
        img = self.run(image)
        img2 = self.run(image_sec)
        img, lb = self.mixup(img, label, img2, label_sec)
        return img, lb
