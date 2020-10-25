import os
import re
import sys
import torch
import pandas as pd
import numpy as np
import progressbar as pb

sys.path.append("..")
from init import DataReader
from model.resnet.resnet import resnet50, resnet_face50
from utils.misc import write_feat, write_meta
from torch.utils.data import DataLoader


def save_feat(ft, name_list, lim):
    __store = {}
    __feats = []
    ft = ft.cpu()
    for dx in range(lim):
        filepath, filename = os.path.split(name_list[dx])
        loc = re.search(r'[\d]+', filename).span()
        name = filename[0:loc[0]-1]
        idx = int(filename[loc[0]:loc[1]])
        __store[name + '/' + str(idx)] = ft[dx].data.numpy()
        __feats.append(ft[dx].data.numpy())
        # if not os.path.exists(path+name):
        #     os.mkdir(path+name)
        # ftx = ft[dx].data.numpy()
        # ftx = ftx.tolist()
        # pt = pd.DataFrame(data=ftx)
        # pt.to_csv(path+name+'/'+str(idx), mode='w', index=None, header=None)
    return __store, __feats


def get(filepath, data):
    _store = {}
    _feats = []
    # load data
    batch_size = 32
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # load model
    device = torch.device('cuda:0')
    model = resnet_face50().cuda()
    # model = resnet50().cuda()
    checkpoint = torch.load(filepath)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    model.eval()  # DropOut/BN

    # get feat
    print('Calculating Feature Map...')
    Total = (data.len - 1) / batch_size + 1
    widgets = [' ', pb.Percentage(),
               ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
               ' ', pb.Timer(),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pgb = pb.ProgressBar(widgets=widgets, maxval=Total).start()
    for i, (inputs, labels, names) in enumerate(data_loader):
        feat = model(inputs.to(device))
        res = save_feat(feat, names, labels.size(0))
        _store.update(res[0])
        _feats += res[1]
        pgb.update(i)
    pgb.finish()
    print('epoch: %d\niters: %d\nloss: %.3lf\ntrain_acc: %.3lf' %
          (checkpoint['epoch'],
           checkpoint['iter'],
           checkpoint['loss'],
           checkpoint['acc']
           ))
    return _store, _feats


if __name__ == '__main__':
    ft_path = '/dev/shm/DATA/wf-mtcnn-vgg16/features.bin'
    labels_path = '/dev/shm/DATA/wf-mtcnn-vgg16/labels.meta'
    # print('Feature saved to %s' % ft_path)
    store, feats = get()
    feats = np.array(feats)
    write_feat(ft_path, feats)
    print('Label saved to %s' % labels_path)
    # lbs = data.label.astype(int).tolist()
    # pt = pd.DataFrame(data=lbs)
    # pt.to_csv(labels_path, mode='w', index=None, header=None)
    # print(lbs, file=f)
    # write_meta(labels_path, lbs)
