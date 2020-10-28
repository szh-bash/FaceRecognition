import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
from utils.DataHandler import Augment, MinS, MaxS, W, H
from config import dataPath
# import utils.mtcnn_simple as mts
# lfw: 5749, 13233
# webface: 10575, 494414
# clean-webface: 10575, 455594

aug = Augment()


class DataReader(Dataset):
    rng = np.random

    def __init__(self, st, data_name):
        self.st = st
        self.data_name = data_name
        filepath = None
        if data_name in dataPath:
            filepath = dataPath[data_name]
        else:
            print('Data Type 404')
            exit(-1)
        path_dir = os.listdir(filepath)
        print('data path:', filepath)
        self.dataset = []
        self.label = []
        self.name = []
        self.person = 0
        self.rng = np.random
        self.idx = []
        self.feat = []
        self.len = 0
        widgets = ['Data Loading: ', pb.Percentage(),
                   ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
                   ' ', pb.Timer(),
                   ' ', pb.ETA(),
                   ' ', pb.FileTransferSpeed()]
        pgb = pb.ProgressBar(widgets=widgets,
                             maxval=494414 if 'Web' in data_name else 13233).start()
        if self.st == 'train':
            self.person = -1
            file = open('/data/shenzhonghai/WebFace-cleaned_list.txt')
            for st in file.readlines():
                st = st.split(' ')
                if int(st[1]) > self.person:
                    self.idx.append(self.len)
                    self.person += 1
                st = st[0].split('\\')
                self.len += 1
                if os.path.exists(filepath+'/'+st[0]+'/'+st[1]):
                    self.name.append(filepath+'/'+st[0]+'/'+st[1])
                    self.label.append(self.person)
                pgb.update(self.len)
            self.person += 1
        else:
            for allDir in path_dir:
                child = os.path.join('%s/%s' % (filepath, allDir))
                child_dir = os.listdir(child)
                self.idx.append(self.len)
                tmp = 0
                for allSon in child_dir:
                    son = os.path.join('%s/%s' % (child, allSon))
                    if self.st == 'test':
                        self.dataset.append(cv2.imread(son))
                    elif self.st == 'feat':
                        cup = []
                        fp = open(son)
                        for st in fp:
                            cup.append(float(st))
                        self.feat.append(cup)
                    self.label.append(self.person)
                    self.name.append(son)
                    pgb.update(self.len)
                    self.len += 1
                    tmp += 1
                self.person += 1
        pgb.finish()
        print('Data mode: '+self.st)
        if self.st == 'feat':
            self.feat = np.array(self.feat, dtype=float)
        self.label = np.array(self.label)
        self.len = self.label.shape[0]
        print('Types:', self.person)
        print('Label:', self.label.shape)
        print('Label_value:', self.label[345:350])
        self.y = torch.from_numpy(self.label).long()

    def __getitem__(self, index):
        if self.st == 'train':
            index_sec = int(self.rng.rand()*self.len)
            # print(index_sec) # same in diff gpu
            label = np.zeros(self.person)
            label_sec = np.zeros(self.person)
            label[self.y[index]] = 1
            label_sec[self.y[index_sec]] = 1
            image = np.array(cv2.imread(self.name[index]), dtype=float).copy()
            image_sec = np.array(cv2.imread(self.name[index_sec]), dtype=float).copy()
            image, label = aug.run2(image, label, image_sec, label_sec)
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            return image, label
        elif self.st == 'test':
            size = (MinS + MaxS) // 2
            idx = (size - W) // 2
            image = np.array(self.dataset[index], dtype=float).copy()
            image = cv2.resize(image, (size, size))
            image = image[idx:idx + H, idx:idx + W, :]
            image = np.transpose(image, [2, 0, 1])
            image = torch.from_numpy((image - 127.5) / 128).float()
            label = np.zeros(self.len)
            label[self.y[index]] = 1
            label = torch.from_numpy(label).float()
            return image, label, self.name[index]
        else:
            exit(-1)

    def __len__(self):
        return self.len
