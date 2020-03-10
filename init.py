import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import progressbar as pb
from utils.DataHandler import Augment
from config import lfwPath, lfwDfPath, webPath, mtWebPath, mtLfwPath
# import utils.mtcnn_simple as mts
# lfw: 5749, 13233
# webface: 10575, 494414
# clean-webface: 10575, 455594


aug = Augment()


class DataReader(Dataset):

    def __init__(self, st, data_name):
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
            elif data_name == 'mtLfw':
                filepath = mtLfwPath
        path_dir = os.listdir(filepath)
        print('data path:', filepath)
        self.dataset = []
        self.label = []
        self.name = []
        self.person = 0
        self.rng = np.random
        # nums = []
        self.idx = []
        self.feat = []
        self.len = 0
        fail = 0
        widgets = ['Data Loading: ', pb.Percentage(),
                   ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
                   ' ', pb.Timer(),
                   ' ', pb.ETA(),
                   ' ', pb.FileTransferSpeed()]
        pgb = pb.ProgressBar(widgets=widgets,
                             maxval=455594 if data_name == 'webFace' or data_name == 'mtWebFace' else 13233).start()
        if self.st == 'train':
            self.person = -1
            file = open('/dev/shm/cleaned_list.txt')
            for st in file.readlines():
                st = st.split(' ')
                if int(st[1]) > self.person:
                    self.idx.append(self.len)
                    self.person += 1
                st = st[0].split('\\')
                self.len += 1
                self.name.append(mtWebPath+'/'+st[0]+'/'+st[1])
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
                    # elif self.st == 'mtcnn':
                    #     if not(os.path.exists(buildPath+'/'+allDir)):
                    #         os.mkdir(buildPath+'/'+allDir)
                    #     dst = os.path.join('%s/%s/%s' % (buildPath, allDir, allSon))
                    #     if mts.run(son, dst) < 0:
                    #         if mts.run(dst, dst) < 0:
                    #             fail += 1

                    self.label.append(self.person)
                    self.name.append(son)
                    pgb.update(self.len)
                    self.len += 1
                    tmp += 1
                # nums.append(tmp)
                self.person += 1
                # if person == 1000:
                #     break
        pgb.finish()
        # print(np.sort(np.array(nums))[-20:])
        if self.st == 'test':
            self.dataset = np.transpose(np.array(self.dataset, dtype=float), [0, 3, 1, 2])
            print(self.dataset.shape)
            self.x = (torch.FloatTensor(self.dataset) - 127.5) / 128.0
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
            x = (250-224) // 2
            y = (250-224) // 2
            return self.x[index, :, x:x + 224, y:y + 224], self.y[index], self.name[index]
        else:
            exit(-1)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    # buildPath = mtLfwPath
    # data = DataReader('mtcnn', 'lfw')
    # data = DataReader('test', 'mtLfw')
    # img = data.__getitem__(0)[0].numpy()
    img = np.array(cv2.imread(mtWebPath+'/0000188/183.jpg'), dtype=float)
    y = 144
    x = (250 - y) // 2
    img = img[x:x+y, x:x+y, :]
    cv2.imwrite('crop.jpg', img)
