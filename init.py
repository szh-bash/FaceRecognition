import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
# person 5749 Total 13233 (lfw)

filePath = "/data/shenzhonghai/lfw/lfw"
pathDir = os.listdir(filePath)
print('data path:', filePath)


class DataReader(Dataset):
    dataset = []
    label = []
    name = []
    person = 0
    rng = np.random
    widgets = ['Loading: ', pb.Percentage(),
               ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
               ' ', pb.Timer(),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pgb = pb.ProgressBar(widgets=widgets, maxval=5749).start()
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filePath, allDir))
        childDir = os.listdir(child)
        for allSon in childDir:
            son = os.path.join('%s/%s' % (child, allSon))
            dataset.append(cv2.imread(son))
            label.append(person)
            name.append(son)
        pgb.update(person)
        person += 1
        # if person == 1000:
        #     break
    pgb.finish()
    print('Data Loaded!')

    dataset = np.transpose(np.array(dataset, dtype=float), [0, 3, 1, 2])
    label = np.array(label)
    print('data:', dataset.shape)
    print('label:', label.shape)
    print('label_value:', label[:5])
    print('Data Shape Transposed!')

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
