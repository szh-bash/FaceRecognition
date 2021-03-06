import os
import re
import sys
import torch
import numpy as np
import progressbar as pb

sys.path.append("..")
from model.resnet import resnet_face50
from utils.misc import write_feat
from torch.utils.data import DataLoader
from config import test_batch_size as batch_size


def save_feat(ft, name_list, lim, md):
    __store = {}
    __feats = []
    __sps = {}
    ft = ft.cpu()
    for dx in range(lim):
        filepath, filename = os.path.split(name_list[dx])

        if md == 0:
            # lfw
            loc = re.search(r'[\d]+', filename).span()
            name = filename[0:loc[0]-1]
            idx = int(filename[loc[0]:loc[1]])
        elif md == 1:
            # faces/grimace
            filpg, name = os.path.split(filepath)
            st = filename.split('.')
            idx = int(st[1])
        else:
            # PIE
            filpg, name = os.path.split(filepath)
            idx = int(filename.split('.')[0])

        if name in __sps:
            __sps[name].append(name + '/' + str(idx))
        else:
            __sps[name] = [name + '/' + str(idx)]
        __store[name + '/' + str(idx)] = ft[dx].data.numpy()
        __feats.append(ft[dx].data.numpy())
        # if not os.path.exists(path+name):
        #     os.mkdir(path+name)
        # ftx = ft[dx].data.numpy()
        # ftx = ftx.tolist()
        # pt = pd.DataFrame(data=ftx)
        # pt.to_csv(path+name+'/'+str(idx), mode='w', index=None, header=None)
    return __store, __feats, __sps


def get(filepath, data):
    _store = {}
    _feats = []
    _sps = {}
    dataname = data.data_name
    print(dataname)
    md = 0 if 'Lfw' in dataname else 2 if 'pie' in dataname else 1
    # load data
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # load model
    device = torch.device('cuda:0')
    model = resnet_face50(test=True).cuda()
    # model = resnet50().cuda()
    checkpoint = torch.load(filepath)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    model.eval()  # DropOut/BN
    print('epoch: %d\niters: %d\nloss: %.3lf\ntrain_acc: %.3lf' %
          (checkpoint['epoch'],
           checkpoint['iter'],
           checkpoint['loss'],
           checkpoint['acc'],
           ))

    # get feat
    print('Calculating Feature Map...')
    Total = (data.len - 1) / batch_size + 1
    widgets = [' ', pb.Percentage(),
               ' ', pb.Bar(marker='>', left='[', right=']', fill='='),
               ' ', pb.Timer(),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pgb = pb.ProgressBar(widgets=widgets, maxval=Total).start()
    for i, (inputs, inputs_flip, labels, names) in enumerate(data_loader):
        feat = model(inputs.to(device))
        feat_flip = model(inputs_flip.to(device))
        feat = (feat+feat_flip)/2
        __store, __feats, __sps = save_feat(feat, names, labels.size(0), md)
        _store.update(__store)
        _feats += __feats
        for x in __sps:
            if x in _sps:
                _sps[x] += __sps[x]
            else:
                _sps[x] = __sps[x]
        pgb.update(i)
    pgb.finish()
    # print(checkpoint['arc'])
    return _store, _feats, _sps


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
