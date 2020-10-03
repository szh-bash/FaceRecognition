import cv2
import numpy as np
# import torch
import torchvision.transforms as trans


_CH = 32
_CW = 32
W = 224
H = 224
MinS = 224
MaxS = 256


class Augment:
    rng = np.random

    def cutout(self, img):
        if self.rng.rand() < 0.5:
            y = int(self.rng.rand() * H) + 1
            x = int(self.rng.rand() * W) + 1
            img[y:min(H, y+_CH), x:min(W, x+_CW), :] = 255.
        return img

    def resize(self, img):
        new_size = self.rng.randint(MinS, MaxS+1)
        img = cv2.resize(img, (new_size, new_size))
        return img

    def crop(self, img):
        dh = img.shape[1] - H
        dw = img.shape[0] - W
        y = int(self.rng.rand()*dh)
        x = int(self.rng.rand()*dw)
        return img[y:y+H, x:x+W, :]

    def rotate(self, img):
        if self.rng.rand() < 0.3:
            img = trans.RandomRotation(img, 15)  # -15 -> +15
        return img

    def flip(self, img):
        if self.rng.rand() < 0.5:
            img = img[:, ::-1, :]
        return img

    def gaussian_blur(self, img):
        if self.rng.rand() < 0.3:
            img = cv2.blur(img, (5, 5))
        return img

    def run(self, img, label):
        img = cv2.resize(img, (H, W))
        img = self.resize(img)
        # img = self.rotate(img)
        img = self.crop(img)
        img = self.gaussian_blur(img)
        img = self.flip(img)
        img = np.transpose(img, [2, 0, 1])
        img = (img - 127.5) / 128.0
        return img, label
