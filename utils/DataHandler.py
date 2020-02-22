import numpy as np
import torch

WO = 250
HO = 250
W = 222
H = 222
DW = WO - W
DH = HO - H


class Augment:
    rng = np.random

    def crop(self, img):
        x = int(self.rng.rand()*DW)
        y = int(self.rng.rand()*DH)
        return img[:, x:x+W, y:y+H]

    def flip(self, img):
        if self.rng.rand() > 0.5:
            # return img[:, :, ::-1]
            torch.flip(img, (0,))
        return img

    def run(self, img, label):
        img = self.crop(img)
        img = self.flip(img)
        # img = (img - 127.5) / 128.0
        img = img / 255.
        return img, label


class DataPipe:
    rng = np.random
    aug = Augment()

    def __init__(self, dataset, label, batch_size):
        # self.dataset = (dataset - 127.5) / 128.0
        self.dataset = dataset / 255.0
        self.label = label
        self.data_total = dataset.shape[0]
        self.batch_size = batch_size
        print("Data Loaded!")

    def get_batch(self):
        index = self.rng.randint(0, self.data_total, self.batch_size)
        data, label = self.aug.run(self.dataset[index], self.label[index])
        return torch.FloatTensor(data), torch.LongTensor(label)

