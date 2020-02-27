import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
from utils.DataHandler import Augment
from config import lfwPath, lfwDfPath, webPath, featPath, mtWebPath
# import utils.mtcnn_simple as mts
# lfw: 5749, 13233
# webface: 10575, 494414


aug = Augment()


class DataReader(Dataset):

    def __init__(self, st, data_name, filepath=featPath):
        self.st = st
        self.data_name = data_name
        if st != 'feat':
            if data_name == 'lfw':
                filepath = lfwPath
            elif data_name == 'lfwDf':
                filepath = lfwDfPath
            elif data_name == 'webFace':
                filepath = webPath
            elif data_name == 'mtWebFace':
                filepath = mtWebPath
        path_dir = os.listdir(filepath)
        print('data path:', filepath)
        self.dataset = []
        self.label = []
        self.name = []
        self.person = 0
        self.rng = np.random
        nums = []
        self.idx = []
        self.feat = []
        self.len = 0
        fail = 0
        widgets = ['Loading: ', pb.Percentage(),
                   ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
                   ' ', pb.Timer(),
                   ' ', pb.ETA(),
                   ' ', pb.FileTransferSpeed()]
        pgb = pb.ProgressBar(widgets=widgets, maxval=494414 if data_name == 'webFace' or data_name == 'mtWebFace'
        else 13233).start()
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
                # elif self.st == 'mtcnn':
                #     if not(os.path.exists(mtWebPath+'/'+allDir)):
                #         os.mkdir(mtWebPath+'/'+allDir)
                #     dst = os.path.join('%s/%s/%s' % (mtWebPath, allDir, allSon))
                #     if mts.run(son, dst) < 0:
                #         if mts.run(dst, dst) < 0:
                #             fail += 1

                self.label.append(self.person)
                self.name.append(son)
                pgb.update(self.len)
                self.len += 1
                tmp += 1
            nums.append(tmp)
            self.person += 1
            # if person == 1000:
            #     break
        pgb.finish()
        print('Data Loaded!')
        print(np.sort(np.array(nums))[-20:])
        if self.st == 'test':
            self.dataset = np.transpose(np.array(self.dataset, dtype=float), [0, 3, 1, 2])
            print(self.dataset.shape)
            self.x = (torch.FloatTensor(self.dataset) - 127.5) / 128.0
            # self.x = torch.FloatTensor(self.dataset) / 255.
        elif self.st == 'feat':
            self.feat = np.array(self.feat, dtype=float)
        self.label = np.array(self.label)
        print('Types:', self.person)
        print('Label:', self.label.shape)
        print('Label_value:', self.label[345:350])
        self.y = torch.LongTensor(self.label)
        if self.st == 'mtcnn':
            print(fail)

    def __getitem__(self, index):
        if self.st == 'train':
            img = torch.FloatTensor(np.transpose(np.array(cv2.imread(self.name[index]), dtype=float), [2, 0, 1]))
            return aug.run(img, self.y[index])
        elif self.st == 'test':
            x = (250-222) // 2
            y = (250-222) // 2
            return self.x[index, :, x:x + 222, y:y + 222], self.y[index], self.name[index]
        else:
            exit(-1)

    def __len__(self):
        return self.len


# if __name__ == '__main__':
    # data = DataReader('mtcnn', 'webFace')
