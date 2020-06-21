import cv2
import numpy as np
# import torch
import torchvision.transforms as trans


W = 224
H = 224
MinS = 256
MaxS = 512


class Augment:
    rng = np.random

    def resize(self, img):
        # new_size = self.rng.randint(MinS, MaxS+1)
        new_size = 384
        new_size = (new_size, new_size)
        img = cv2.resize(img, new_size)
        return img

    def crop(self, img):
        dw = img.shape[0] - W
        dh = img.shape[1] - H
        x = int(self.rng.rand()*dw)
        y = int(self.rng.rand()*dh)
        return img[x:x+W, y:y+H, :]

    def rotate(self, img):
        if self.rng.rand() < 0.3:
            img = trans.RandomRotation(img, 15)  # -15 -> +15
        return img

    def flip(self, img):
        if self.rng.rand() < 0.5:
            img = img[:, ::-1, :]
            # img = torch.flip(img, (2,))
        return img

    def run(self, img, label):
        # img = self.resize(img)
        # img = self.rotate(img)
        img = self.flip(img)
        img = self.crop(img)
        img = np.transpose(img, [2, 0, 1])
        img = (img - 127.5) / 128.0
        return img, label


# class DataPipe:
#     rng = np.random
#     aug = Augment()
#
#     def __init__(self, dataset, label, batch_size):
#         # self.dataset = (dataset - 127.5) / 128.0
#         self.dataset = dataset / 255.0
#         self.label = label
#         self.data_total = dataset.shape[0]
#         self.batch_size = batch_size
#         print("Data Loaded!")
#
#     def get_batch(self):
#         index = self.rng.randint(0, self.data_total, self.batch_size)
#         data, label = self.aug.run(self.dataset[index], self.label[index])
#         return torch.from_numpy(data), torch.from_numpy(label)
