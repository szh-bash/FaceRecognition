import cv2
import os
import torch
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
# person 5749 Total 13233 (lfw)

# filePath = "/data/shenzhonghai/lfw/lfw-deepfunneled"
filePath = "/data/shenzhonghai/lfw/lfw"
pathDir = os.listdir(filePath)
dataset = []
label = []
name = []
person = 0
print('data path:', filePath)


class DataReader(Dataset):

    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filePath, allDir))
        childDir = os.listdir(child)
        name.append(allDir)
        for allSon in childDir:
            son = os.path.join('%s/%s' % (child, allSon))
            dataset.append(cv2.imread(son))
            label.append(person)
        person += 1
        if person % 500 == 0:
            print(person)
        # if person == 1000:
        #     break

    dataset = np.transpose(np.array(dataset, dtype=float), [0, 3, 1, 2])
    label = np.array(label)
    print('data:', dataset.shape)
    print('label:', label.shape)
    print(label[:5])
    rng = np.random

    def __init__(self):
        self.x = torch.FloatTensor(self.dataset)
        self.y = torch.LongTensor(self.label)
        self.len = self.dataset.shape[0]

    def __getitem__(self, index):
        x = int(self.rng.rand() * (250-222))
        y = int(self.rng.rand() * (250-222))
        return self.x[index, :, x:x + 222, y:y + 222], self.y[index]

    def __len__(self):
        return self.len
