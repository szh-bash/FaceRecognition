import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
# lfw: 5749, 13233
# webface: 10575, 494414

lfwPath = '/dev/shm/lfw'
webfacePath = '/dev/shm/CASIA-WebFace'


class DataReader(Dataset):

    def __init__(self, st, data_name):
        self.st = st
        self.data_name = data_name
        filepath = lfwPath if data_name == 'lfw' else webfacePath
        path_dir = os.listdir(filepath)
        print('data path:', filepath)
        self.dataset = []
        self.label = []
        self.name = []
        self.person = 0
        self.rng = np.random
        widgets = ['Loading: ', pb.Percentage(),
                   ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
                   ' ', pb.Timer(),
                   ' ', pb.ETA(),
                   ' ', pb.FileTransferSpeed()]
        pgb = pb.ProgressBar(widgets=widgets, maxval=13233 if data_name == 'lfw' else 494414).start()
        self.len = 0
        for allDir in path_dir:
            child = os.path.join('%s/%s' % (filepath, allDir))
            child_dir = os.listdir(child)
            for allSon in child_dir:
                son = os.path.join('%s/%s' % (child, allSon))
                if self.st == 'test':
                    self.dataset.append(cv2.imread(son))
                self.label.append(self.person)
                self.name.append(son)
                pgb.update(self.len)
                self.len += 1
            self.person += 1
            # if person == 1000:
            #     break
        pgb.finish()
        print('Data Loaded!')
        if self.st == 'test':
            self.dataset = np.transpose(np.array(self.dataset, dtype=float), [0, 3, 1, 2])
            print('Img:', self.dataset.shape)
            self.x = (torch.FloatTensor(self.dataset) - 127.5) / 128.0
        self.label = np.array(self.label)
        print('Label:', self.label.shape)
        print('Label_value:', self.label[345:350])
        self.y = torch.LongTensor(self.label)

    def __getitem__(self, index):
        x = int(self.rng.rand() * (250-222))
        y = int(self.rng.rand() * (250-222))
        if self.st == 'train':
            img = (torch.FloatTensor(np.transpose(np.array(cv2.imread(self.name[index]), dtype=float), [2, 0, 1])) - 127.5) / 128.0
            return img[:, x:x + 222, y:y + 222], self.y[index]
        elif self.st == 'test':
            return self.x[index, :, x:x + 222, y:y + 222], self.y[index], self.name[index]
        else:
            exit(-1)

    def __len__(self):
        return self.len
